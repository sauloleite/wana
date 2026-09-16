# wana: plano de construção

> *wana* é "Caminho" em WANYAM, língua indígena extinta pertencente à família linguística Txapacura. A biblioteca é o "caminho" entre os datasets de fine-tuning e a evidência de por que cada exemplo ficou ou saiu.

Caso queira analisar melhor a lib wyra, ela se encontra aqui:/Users/sauloleite/Documents/GitHub/wyra

## 1. Escopo

**Problema.** Entre `wyra build` (ou qualquer JSONL de SFT) e o treinador (TRL, Axolotl, Unsloth, Azure OpenAI) existe uma etapa feita hoje com scripts soltos: decidir quais exemplos valem o treino e provar que o treino não vaza no conjunto de avaliação.

**Entrada.** Um ou mais JSONL nos formatos `openai-chat`, `alpaca` ou `sharegpt`, opcionalmente com o `manifest.json` do wyra ao lado. Conjuntos de avaliação no mesmo formato (o `valid.jsonl` do wyra, um benchmark baixado, um golden set interno).

**Saída.**

| Artefato | Conteúdo |
|---|---|
| `scored.jsonl` | cada exemplo original mais um bloco `wana` com os scores |
| `selected.jsonl` | o subset escolhido, na ordem original do arquivo |
| `contamination.json` | pares (treino, avaliação) suspeitos, com nível e evidência |
| `manifest.json` | proveniência encadeada: hashes de entrada e saída, parâmetros, versão, `parent` |
| relatório em terminal / Markdown | resumo legível para revisão humana |

**Não faz.** Não gera dados, não treina, não avalia modelo. Não substitui leitura de amostra: um exemplo errado e difícil pontua alto no IFD.

## 2. Princípios (herdados do wyra)

1. **Determinismo primeiro.** Tudo que não precisa de modelo é código puro: n-gramas, MinHash, seleção, manifesto. Mesma entrada, mesma seed, mesmo resultado, em qualquer máquina.
2. **Zero dependências no core.** `pip install wana` instala só stdlib. Modelos e embeddings entram por extras.
3. **Portas como `typing.Protocol`.** Cada seam é substituível sem herança; a versão `Fake` de cada porta permite testar sem rede e sem pesos.
4. **CI como cidadão de primeira classe.** Todo comando devolve exit code significativo e escreve JSON estável.
5. **Configuração pelo ambiente e pela linha de comando.** Nenhum arquivo de configuração próprio; nenhuma credencial em código.
6. **O manifesto é subproduto, não papelada.** Ele nasce do processamento, não de um passo extra.

## 3. Arquitetura

Arquitetura limpa em quatro anéis. A regra de dependência é única: importações apontam para dentro. `domain` não importa nada de fora; `adapters` e `cli` conhecem `application`; `application` conhece só `domain` e `ports`.

```
wana/
├── domain/            # entidades e funções puras (anel interno)
│   ├── example.py     # Example, Message, Role, ExampleId
│   ├── score.py       # Score, ScoreSet, ScoreName
│   ├── selection.py   # Selection, Budget, Decision(KEEP|DROP, reason)
│   ├── contamination.py  # Hit, Level(EXACT|NEAR|SEMANTIC), Report
│   └── manifest.py    # Manifest, Step, Digest
├── ports/             # Protocols (contratos)
│   ├── reader.py      # ExampleReader
│   ├── writer.py      # ExampleWriter, ReportWriter
│   ├── scorer.py      # Scorer
│   ├── selector.py    # Selector
│   ├── matcher.py     # Matcher (contaminação)
│   ├── logprob.py     # LogProbProvider
│   ├── embedder.py    # Embedder
│   └── tokens.py      # TokenCounter
├── application/       # casos de uso (orquestração, sem I/O direto)
│   ├── score_dataset.py
│   ├── select_subset.py
│   ├── check_contamination.py
│   ├── build_manifest.py
│   └── pipeline.py    # compõe os quatro acima; é o que a CLI chama
├── adapters/          # implementações concretas das portas
│   ├── io/            # jsonl (+ .gz/.xz/.bz2), formatos openai-chat/alpaca/sharegpt
│   ├── scoring/       # ifd.py, length.py, diversity.py
│   ├── selection/     # topk.py, stratified.py, diverse_greedy.py
│   ├── matching/      # ngram.py, minhash.py, embedding.py
│   ├── logprob/       # transformers_cpu.py, openai_compat.py, fake.py
│   ├── embedding/     # hashing.py (zero-dep), onnx.py, fake.py
│   └── report/        # terminal.py, markdown.py, json.py
├── cli/               # argparse; traduz flags em chamadas de application
│   └── main.py
├── registry.py        # registro por nome de scorers, selectors, matchers
└── __init__.py        # API pública re-exportada
```

### Mapeamento SOLID

| Princípio | Onde aparece |
|---|---|
| **S** | um módulo por responsabilidade: `ifd.py` só calcula IFD; `minhash.py` só compara; o manifesto é montado em um único lugar |
| **O** | novo scorer, selector ou matcher entra por `registry.register(...)` implementando a porta; nenhum arquivo existente muda |
| **L** | `FakeLogProbProvider` e `FakeEmbedder` cumprem exatamente os mesmos contratos; a suíte de testes roda inteira sobre eles |
| **I** | portas pequenas: `Scorer.score(example) -> Score`, `Matcher.hits(train, eval) -> Iterable[Hit]`; ninguém implementa método que não usa |
| **D** | casos de uso recebem portas por construtor; `cli` e `pipeline` são o único ponto de montagem (composition root) |

### KISS, na prática

- `argparse`, não Click nem Typer. Uma dependência a menos e a CLI do wyra já é assim.
- Processamento em streaming, um exemplo por vez, sem `pandas`, sem `datasets` no core.
- Sem async, sem plugins descobertos por entry point, sem YAML. Se a montagem cabe em uma função, ela fica em uma função.
- Um único objeto `Example` atravessa toda a pipeline; formatos são convertidos na borda (leitura e escrita), nunca no meio.

## 4. Casos de uso e algoritmos

### 4.1 `score`: pontuação por exemplo

**IFD (Li et al., 2023).** Para um par instrução Q e resposta A:

- `s(A|Q)`: perda média por token de A condicionada a Q (com o chat template do modelo de scoring).
- `s(A)`: perda média por token de A sem instrução.
- `IFD = s(A|Q) / s(A)`.

Interpretação: perto de 1, a instrução não influencia a resposta; quanto maior, mais a instrução "ensina". Valores acima de 1 indicam desalinhamento (a instrução atrapalha) e são marcados, seguindo o paper. Superfiltering (Li et al., 2024) sustenta calcular com um modelo pequeno: o ranking se mantém entre modelos.

Implementação: `LogProbProvider.mean_nll(prefix, target) -> float`. O adapter `transformers_cpu` faz duas passagens forward por exemplo, sem geração. O adapter `openai_compat` usa `logprobs` quando o servidor expõe (vLLM, LM Studio). `fake` devolve valores fixos para teste.

Scores auxiliares, sem modelo: comprimento em tokens da resposta, razão resposta/instrução, contagem de turnos. Todos vão para o bloco `wana.scores` de cada linha.

### 4.2 `select`: escolha do subset

- `topk`: ordena por um score e corta pelo orçamento (`--keep 0.2` ou `--keep 5000`).
- `stratified`: aplica o corte dentro de estratos (`--by source`, `--by turns`) para preservar a distribuição de origem.
- `diverse_greedy` (inspirado em DEITA): percorre em ordem de score e aceita um exemplo apenas se a distância ao mais próximo já aceito exceder um limiar. A distância vem de `Embedder`; sem extra instalado, o `hashing` embedder (TF-IDF por hashing, stdlib) garante que o comando funciona.

A decisão de cada exemplo (KEEP ou DROP com motivo) é registrada: `{"decision": "DROP", "reason": "ifd_above_1"}`.

### 4.3 `check`: contaminação treino × avaliação

Três níveis, do mais barato ao mais caro, sempre nessa ordem:

| Nível | Método | Padrão | Referência |
|---|---|---|---|
| EXACT | sobreposição de 13-gramas | ≥ 1 n-grama compartilhado | GPT-3 (Brown et al., 2020) usa 13-gramas; GPT-4 usa 50 caracteres |
| NEAR | MinHash + LSH sobre shingles | Jaccard ≥ 0.8 | mesmo mecanismo do `curation.NearDuplicate` do wyra |
| SEMANTIC | cosseno entre embeddings | ≥ 0.9 (opt-in, extra) | Yang et al. (2023): paráfrases e traduções escapam do n-grama |

Cada `Hit` guarda o par de ids, o nível, o valor e o trecho de evidência. O exit code é 1 se houver hit EXACT ou NEAR (configurável com `--fail-on`).

Cuidado específico do wyra: o `TemplateGenerator` gera frases quase idênticas por construção. O matcher recebe um `--ignore-template` que remove o texto do template antes de comparar, para não confundir moldura com conteúdo.

### 4.4 `manifest`: proveniência encadeada

```json
{
  "wana_version": "0.1.0",
  "created_at": "2026-10-01T12:00:00Z",
  "parent": {"path": "dataset/manifest.json", "sha256": "9c1a…", "tool": "wyra 0.1.1"},
  "inputs": [{"path": "dataset/train.jsonl", "sha256": "…", "records": 15000}],
  "eval_sets": [{"path": "dataset/valid.jsonl", "sha256": "…", "records": 1500}],
  "steps": [
    {"name": "score", "scorer": "ifd", "model": "Qwen/Qwen2.5-0.5B", "revision": "abc123", "prompt_sha256": "…"},
    {"name": "select", "selector": "diverse_greedy", "keep": 0.2, "threshold": 0.9, "seed": 42, "in": 15000, "out": 3000},
    {"name": "check", "matchers": ["ngram13", "minhash"], "hits": {"EXACT": 0, "NEAR": 2, "SEMANTIC": null}}
  ],
  "outputs": [{"path": "selected/train.jsonl", "sha256": "…", "records": 3000}]
}
```

`parent` é opcional: sem wyra, a cadeia começa aqui. Dois manifestos com os mesmos `inputs`, `steps` e `seed` produzem os mesmos `outputs`; isso é testado.

## 5. Interface

### CLI

```
wana check  train.jsonl --eval valid.jsonl --eval mmlu.jsonl
wana score  train.jsonl -o scored.jsonl --scorer ifd --model qwen2.5-0.5b
wana select scored.jsonl -o selected/ --by ifd --keep 0.2 --diverse
wana run    dataset/ -o selected/ --eval valid.jsonl --keep 0.2   # os três em sequência
wana explain selected/manifest.json                               # relatório legível
```

Variáveis de ambiente: `URUPEMA_PROVIDER`, `URUPEMA_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `URUPEMA_CACHE_DIR`. Mesmos nomes e semântica do wyra, trocando o prefixo.

### Python

```python
from wana import check_contamination, score_dataset, select_subset, run

report = check_contamination("train.jsonl", eval_sets=["valid.jsonl"])
assert report.ok

run("dataset/", out_dir="selected", eval_sets=["dataset/valid.jsonl"], keep=0.2, seed=42)
```

Toda função pública aceita caminhos ou objetos já carregados, e devolve dataclasses, nunca dicts soltos.

### Extras

```
pip install wana             # core, sem dependências
pip install "wana[score]"    # transformers + torch (CPU) para IFD
pip install "wana[embed]"    # onnxruntime + modelo de embedding pequeno
pip install "wana[tokens]"   # tiktoken
pip install "wana[all]"
```

## 6. Qualidade e engenharia

**Testes.** `pytest`, cobertura mínima 90% no `domain` e `application`. Três camadas:
1. Unitários puros sobre `domain` (sem I/O).
2. Casos de uso com `Fake*` providers (sem rede, sem pesos).
3. Golden tests: fixtures pequenas de JSONL com saída esperada versionada; qualquer mudança de comportamento aparece como diff.

Propriedades verificadas com `hypothesis`: determinismo (mesma entrada e seed → bytes idênticos), idempotência do `select` sobre o próprio resultado, simetria do matcher.

**Tipagem e estilo.** `mypy --strict`, `ruff` (lint e format), `pre-commit`. Docstrings no formato do wyra: dizem o que a função garante, não o que ela faz linha a linha.

**CI (GitHub Actions).** Matriz Python 3.10 a 3.13, Linux/macOS/Windows para o core; um job separado com `[score]` só no Linux. Publicação por Trusted Publishing, com atestação, como no wyra.

**Versionamento.** SemVer. `0.x` até a API pública estabilizar; `CHANGELOG.md` mantido a mão; commits no padrão Conventional Commits.

**Decisões registradas.** Um `docs/adr/` com uma ADR curta por decisão estrutural (por que argparse, por que hashing embedder no core, por que o manifesto não assina criptograficamente na 0.x).

**Definition of Done por funcionalidade.** Testes das três camadas, docstring, entrada no CHANGELOG, exemplo no README, ADR se mudou arquitetura.

## 7. Roadmap

| Fase | Entrega | Critério de aceite |
|---|---|---|
| 0 | Nome reservado no PyPI e npm, esqueleto, CI verde, README com a promessa | `pip install wana==0.0.1` funciona e `wana --version` responde |
| 1 (0.1) | `check` com EXACT e NEAR, leitura dos três formatos, manifesto, relatório terminal | roda em CI contra o `valid.jsonl` do wyra e falha quando há vazamento plantado |
| 2 (0.2) | `score` com IFD via `transformers_cpu` e `fake`; `select` topk e stratified | reproduz a curva do paper de IFD em um dataset público pequeno (Alpaca-1k) |
| 3 (0.3) | `diverse_greedy`, embedder `hashing` e `onnx`, nível SEMANTIC no `check`, relatório Markdown | seleção de 20% supera random em avaliação por juiz num experimento documentado |
| 4 (0.4) | `run` encadeado, `parent` apontando para o manifesto do wyra, `explain` | `wyra build … && wana run …` produz cadeia verificável de dois manifestos |
| 5 (0.5) | porte Node do que é puro: `check` e verificação de manifesto (sem IFD) | mesmo JSON de saída nos dois runtimes, validado por golden tests compartilhados |

Cada fase é publicável sozinha. A ordem coloca primeiro o que não precisa de modelo, para que a lib seja útil desde a 0.1 sem baixar um byte de pesos.

## 8. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| IFD é lento em CPU (duas passagens por exemplo) | modelo padrão de 0.5B, chunk por exemplo, cache de resultados por `sha256(exemplo, modelo, revisão)`; `--resume` |
| IFD depende do modelo de scoring, não do modelo que será treinado | manifesto registra modelo e revisão; documentação explícita de que o score é relativo |
| Falso positivo de contaminação em datasets gerados por template | `--ignore-template`; níveis separados; relatório mostra evidência para revisão humana |
| Extras pesados (`torch`) assustam o usuário do core | core sem dependências; mensagens de erro nomeiam o extra exato a instalar |
| Drift de API do `transformers` | adapter isolado; teste de fumaça semanal em CI contra a versão mais recente |

## 9. Ponte com a tese

O manifesto encadeado é uma instância concreta de proveniência formal para dados de treino (a linha ProvLLM). A afirmação mensurável que a lib permite testar: *"um subset de X% selecionado por IFD e diversidade, com contaminação zero contra o conjunto de avaliação, iguala ou supera o conjunto completo na avaliação Y"*, reprodutível por terceiros a partir dos manifestos. O experimento da fase 3 é o primeiro resultado publicável.

## 10. Primeiros passos (esta semana)

1. Reservar `wana` no PyPI e no npm.
2. Criar o repositório com `pyproject.toml`, `ruff`, `mypy`, `pytest`, `pre-commit` e o workflow de CI copiado do wyra.
3. Escrever `domain/example.py` e `ports/*.py` antes de qualquer adapter: os contratos primeiro.
4. Implementar `adapters/io/jsonl.py` reaproveitando a lógica de leitura do wyra.
5. Fechar a fase 1 com `check` EXACT e NEAR e o manifesto, e publicar a 0.1.0.
