#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <utility>

static const int num_threads = 4;


double even_odd_sort(int* arr, int n)
{
    bool sorted = false;
    int init = 0;

    double t1 = omp_get_wtime();
    while (!sorted)
    {
        sorted = true;
        for (int i = init; i < n - 1; i += 2)
        {
            if (arr[i] > arr[i + 1])
            {
                std::swap(arr[i], arr[i + 1]);
                sorted = false;
            }
        }
        init = 1 - init;
    }
    double t2 = omp_get_wtime();
    return (t2 - t1) * 1000.0;;
}


double even_odd_sort_omp(int* arr, int n)
{
    bool sorted = false;
    int init = 0;

    double t1 = omp_get_wtime();
    while (!sorted)
    {
        sorted = true;
#pragma omp parallel for
        for (int i = init; i < n - 1; i += 2)
        {
            if (arr[i] > arr[i + 1])
            {
                std::swap(arr[i], arr[i + 1]);
                sorted = false;
            }
        }
        init = 1 - init;
    }
    double t2 = omp_get_wtime();
    return (t2 - t1) * 1000.0;;
}

int* generate_array(int n)
{
    static int cnt;
    cnt++;
    int* res = new int[n];
    srand(cnt);
    for (int i = 0; i < n; ++i)
    {
        res[i] = rand() % 1000;
    }
    return res;
}


// check if array is properly sorted
bool check_array(const int* arr, int n)
{
    for (int i = 0; i < n - 1; ++i)
    {
        if (arr[i] > arr[i + 1])
            return false;
    }
    return true;
}


void test(int n, int cnt)
{
    double t = 0.0;
    for (int i = 0; i < cnt; ++i)
    {
        int* arr = generate_array(n);
        t += even_odd_sort(arr, n);
        if (!check_array(arr, n))
        {
            printf("sort failed\n");
            delete[] arr;
            return;
        }
        delete[] arr;
    }
    printf("array size: %d, average time: %.2lf ms\n", n, t / cnt);
}

void test_omp(int n, int cnt)
{
    double t = 0.0;
    for (int i = 0; i < cnt; ++i)
    {
        int* arr = generate_array(n);
        t += even_odd_sort_omp(arr, n);
        if (!check_array(arr, n))
        {
            printf("omp sort failed\n");
            delete[] arr;
            return;
        }
        delete[] arr;
    }
    printf("array size: %d, average time using omp: %.2lf ms\n", n, t / cnt);
}


int main()
{
    omp_set_num_threads(num_threads);
    test(500, 10);
    test_omp(500, 10);
    test(1000, 10);
    test_omp(1000, 10);
    test(1500, 10);
    test_omp(1500, 10);
    test(2000, 10);
    test_omp(2000, 10);
    test(2500, 10);
    test_omp(2500, 10);

    return 0;
}