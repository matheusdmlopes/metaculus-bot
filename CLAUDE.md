# metaculus-bot

Bot que prevê sozinho as perguntas do torneio de bots da Metaculus (FutureEval) e do MiniBench, com o Claude pela API da Anthropic. Visão geral, comandos e variáveis no `README.md`.

## Regra do torneio: nada de humano no meio

O regulamento proíbe rodar o bot em pergunta aberta (ou que vai abrir) do torneio, ver a saída e mudar o bot por causa dela. Quebrar isso desclassifica. Na prática:

- Não rodar `uv run python main.py` (modo torneio) à mão para "ver o que sai", nem tirar as travas de `--pergunta` e `--sem-publicar`.
- Não ler previsão ou raciocínio de pergunta ainda aberta do torneio para decidir mudança de prompt.
- Para ajustar o bot: área de teste (`--mode test_questions`), `--texto`, `--pergunta` com pergunta de fora do torneio, ou revisão de perguntas já resolvidas (skill `review-bot`, só leitura).

## Repositório público

O código fica em <https://github.com/matheusdmlopes/metaculus-bot>, aberto, e os logs do Actions também. Nada de chave, `.env`, conteúdo de `relatorios/` ou previsão de pergunta ainda aberta em commit, log ou mensagem de commit. As chaves vivem só no `.env` local e nos segredos do repositório.

## Dinheiro

Toda execução de verdade gasta crédito da API (pesquisa com busca na web + cinco previsões por pergunta). Avisar antes de rodar e dizer o custo no fim; o resumo da execução mostra o total. `uv run pytest` é offline e não gasta nada. Os tetos (US$ 3 por execução, US$ 5 por dia) ficam no `Settings` do `main.py`. O Matheus fixou o máximo de US$ 5 por dia em 17/11/2026: o crédito é dividido com outros projetos dele. Não subir teto nem trocar para modelo mais caro sem ele pedir.

## Código

- Chamada ao Claude só pelo `claude_llm.py` (SDK oficial `anthropic`, streaming, `fallbacks="default"`). Não usar o `GeneralLlm`/litellm do forecasting-tools: os padrões dele mandam `temperature`, que os modelos atuais rejeitam.
- Saída estruturada (`schema`) e busca na web (`tools`) não vão na mesma chamada.
- O CI do bot roda em Python 3.12, a mesma versão do ambiente local.
- Depois de mexer: `uv run pytest`.
- `configurar.sh`: chave colada aparece na tela e é conferida (tamanho, formato). Nada de entrada invisível (`read -s`).
