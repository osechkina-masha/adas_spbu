# ADAS algorithms hyperparameters tuning with decision trees

__Цель работы__: Исследовать и разработать инструмент для полуавтоматического подбора гиперпараметров алгоритмов детектирования линий с учетом погодных условий (окружающей среды) на основе деревьев решений

На данный момент доступны методы оптимизации:
1. GeneticLearner
2. HyperOptLearner
3. REINFORCELearner

Чтобы запустить примеры из core/examples достаточно запустить:

```
pip install -r requirements.txt
python -m core.examples.genetic
```