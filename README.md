# High_performance_computing

task1: Вычисление числа Pi  
> integral_0-1( 4.0 / (1 + x^2) )
+ Сначала, написать последовательную программу и проверить, что результат приблизительно равен числу π.
+ Затем применить директиву #pragma omp parallel for. Сравнить полученный результат с результатом выполнения последовательной программы.
+ После этого применить параметр reduction(). Проверить результаты ещё раз.

task2: Разработайте программу решения задачи поиска максимального значения среди мини-
мальных элементов строк матрицы

task3: Провести сравнение времени выполнения последовательной программы относительно указанных вариантов параллельной программы на 
различном количестве нитей (коэффициент ускорения). Размер исходных данных переменный. Результаты измерений представить
в виде таблицы или графика.
