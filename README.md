# Bot de previsão da Metaculus

Bot que responde sozinho às perguntas do [torneio de bots da Metaculus](https://www.metaculus.com/futureeval/) (FutureEval) e do MiniBench, usando o Claude pela API da Anthropic. Partiu do [template oficial](https://github.com/Metaculus/metac-bot-template); a parte do modelo foi reescrita.

## Como funciona

Para cada pergunta aberta:

1. **Pesquisa.** Uma chamada ao Claude com busca na web e leitura de página monta um briefing: situação atual frente ao critério de resolução, taxas históricas, notícias recentes, o que mercados dizem, armadilhas do critério.
2. **Cinco previsões.** Cinco chamadas independentes leem o briefing, cada uma por um ângulo (taxa histórica, cadeia causal, status quo, advogado do diabo, letra miúda do critério) e devolvem o raciocínio e o número.
3. **Agregação e envio.** A biblioteca `forecasting-tools` junta as cinco (mediana), envia a previsão e posta o comentário com a pesquisa e os raciocínios, como a regra do torneio exige.

As perguntas do torneio ficam abertas por cerca de 3 horas e aceitam uma previsão só. Por isso o bot roda no GitHub Actions a cada 20 minutos e pula o que já previu.

## Arquivos

| Arquivo | O que tem |
|---|---|
| `main.py` | O bot (pesquisa, previsões por tipo de pergunta) e a linha de comando |
| `prompts.py` | Os textos enviados ao Claude, os cinco ângulos e os formatos de resposta |
| `claude_llm.py` | Chamada à API da Anthropic: streaming, recusa com fallback, custo, teto de gasto |
| `tests/test_offline.py` | Testes com um cliente falso, sem rede e sem gasto |
| `.github/workflows/` | Execução no torneio a cada 20 min, teste manual e revisão semanal |

## Preparar

```bash
uv sync
./configurar.sh   # assistente: abre cada página, pede as duas chaves, grava o .env e confere
```

Ou à mão: `cp .env.template .env`, preencher as duas chaves e rodar `uv run python main.py --checar`.

As duas chaves:

- `METACULUS_TOKEN`: crie sua conta na Metaculus, depois o bot em <https://www.metaculus.com/accounts/settings/bots/>. O token é o do bot. Preencha também o [formulário de participação](https://forms.gle/aQdYMq9Pisrf1v7d8).
- `ANTHROPIC_API_KEY`: no Console da Anthropic. Vale criar um workspace só para o bot, com limite de gasto mensal.

`--checar` valida as duas e faz duas chamadas mínimas ao Claude (uma delas com busca na web); custa centavos.

## Rodar

```bash
uv run python main.py --texto "Will SpaceX reach Mars orbit before 2030?"   # pergunta solta, não publica
uv run python main.py --pergunta https://www.metaculus.com/questions/1234/  # pergunta do site, não publica
uv run python main.py --mode test_questions --limite 1                      # área de teste, publica lá (não pontua)
uv run python main.py --mode test_questions --sem-publicar                  # área de teste, só mostra
uv run python main.py                                                       # torneio + MiniBench, publica
```

Nos modos locais o relatório completo de cada pergunta fica em `relatorios/`.

## Regra que não pode quebrar

O torneio proíbe humano no meio: não se roda o bot numa pergunta aberta do torneio para olhar a resposta e ajustar o bot em cima dela. Por isso:

- `--pergunta` recusa pergunta do torneio em andamento;
- `--sem-publicar` só existe na área de teste;
- no modo torneio o log mostra só o endereço e o estado de cada pergunta, sem previsão nem raciocínio.

Ajuste de prompt se faz com a área de teste, com perguntas de fora do torneio ou olhando perguntas que já resolveram (workflow `Review recent forecasts`).

## Custo e ajustes

O fim de cada execução mostra chamadas, buscas, dólares gastos e o total do dia. O custo por pergunta também sai no comentário publicado. Variáveis opcionais (no `.env` ou em Settings → Variables do repositório):

| Variável | Padrão | Para quê |
|---|---|---|
| `BOT_RESEARCH_MODEL` | `claude-sonnet-5-5` | Modelo da pesquisa (`claude-opus-5-5` é melhor e custa o dobro) |
| `BOT_FORECAST_MODEL` | `claude-sonnet-5-5` | Modelo das previsões |
| `BOT_RESEARCH_EFFORT`, `BOT_FORECAST_EFFORT` | `high` | Esforço de raciocínio |
| `BOT_MAX_SEARCHES`, `BOT_MAX_FETCHES` | `6`, `3` | Buscas e páginas lidas por pergunta |
| `BOT_FORECASTS_PER_QUESTION` | `5` | Previsões por pergunta |
| `BOT_MAX_USD_PER_RUN` | `3` | Teto por execução; passou, as perguntas restantes ficam para a próxima |
| `BOT_MAX_USD_PER_DAY` | `5` | Teto por dia (UTC), somando todas as execuções; uns US$ 150 por mês no máximo |
| `BOT_PARALLEL_CALLS`, `BOT_PARALLEL_QUESTIONS` | `4`, `1` | Chamadas e perguntas em paralelo |

No teto do dia, pergunta nova só começa se couber: conta o gasto de hoje mais o custo da pergunta mais cara do dia (US$ 1 enquanto não houve nenhuma). O dia só passa do teto se uma pergunta sair mais cara que todas as anteriores. O gasto do dia fica em `.gasto/gasto-do-dia.json`: no Actions ele passa de uma execução para a outra pelo cache do workflow, e na sua máquina é um arquivo à parte. O workflow `Test Bot` não entra na conta do dia. A trava que não depende do bot é o limite mensal do workspace no Console da Anthropic.

## No GitHub Actions

O último passo do `./configurar.sh` manda os segredos `METACULUS_TOKEN` e `ANTHROPIC_API_KEY` para o repositório e liga o agendamento. À mão: criar os dois em Settings → Secrets and variables → Actions e rodar `gh workflow enable run_bot_on_tournament.yaml`.

O workflow `Forecast on new AI tournament questions` roda sozinho a cada 20 minutos; `Test Bot` roda na área de teste quando acionado à mão. Para parar o bot: `gh workflow disable run_bot_on_tournament.yaml`.

Em repositório público o GitHub desliga agendamentos depois de 60 dias sem atividade; o último passo do workflow religa o agendamento pela API a cada execução para isso não acontecer.

## Testes

```bash
uv run pytest
```
