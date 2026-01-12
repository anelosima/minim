# Minimally modified Metaculus forecasting bot
minim is a forecasting bot for the Metaculus AI forecasting benchmark. Previous iterations of the template bots with SOTA models have placed very well in the Metaculus benchmark tournaments. In my own testing, however, various parts of the bot sometimes fail quite dramatically at their intended task. For example:
- the queries which are used to research are sometimes very poor;
- sometimes the AskNews API endpoint handles them very poorly;
- sometimes the news received does not contain up-to-date figures or other information;
- sometimes the reasoning has obvious surface-level flaws (e.g. predicting based on things which may only happen after the relevant parts of the question are already decided)
The aim of minim is to add checks at various points to ensure that everything is working as intended, and to fix the issue if it is not. I estimate that up to about 10% of questions present a problem in the template bot implementation which is resolvable automatically. The goal is to resolve as many of these problems as possible while maintaining that the behaviour on all other questions is exactly the same as it is in the template bot, since the template bot already performs very well.
