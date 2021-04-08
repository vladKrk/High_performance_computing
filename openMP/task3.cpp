#include <omp.h>
#include <cstdio>
#include <iostream>
#include <cstdlib>


static int num_threads = 8;

int* matrix_vector_omp_rows(int** matrix, int* vector, int m, int n, double& dt)
{
	int* res = new int[m];
	double t1 = omp_get_wtime();
#pragma omp parallel for
	for (int i = 0; i < m; ++i)
	{
		int sum = 0;
		for (int j = 0; j < n; ++j)
		{
			sum += matrix[i][j] * vector[j];
		}
		res[i] = sum;
	}
	double t2 = omp_get_wtime();
	dt = (t2 - t1) * 1000.0;
	return res;
}


int* matrix_vector_omp_columns(int** matrix, int* vector, int m, int n, double& dt)
{
	int* res = new int[m];
	for (int i = 0; i < m; ++i)
	{
		res[i] = 0;
	}
	double t1 = omp_get_wtime();
#pragma omp parallel for
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
#pragma omp atomic
			res[j] += matrix[j][i] * vector[i];
		}
	}
	double t2 = omp_get_wtime();
	dt = (t2 - t1) * 1000.0;
	return res;
}


void block_vec_mult(int** matrix, const int* vector, int* res, int m, int n, int block_index)
{
	int block_row_size = m / num_threads * 2;
	int block_col_size = n / 2;
	int block_row_start = block_index / 2;
	int block_col_start = block_index % 2;

	for (int i = block_row_start * block_row_size; i < (block_row_start + 1) * block_row_size; ++i)
	{
		int sum = 0;

		for (int j = block_col_start * block_col_size; j < (block_col_start + 1) * block_col_size; ++j)
		{
			sum += matrix[i][j] * vector[j];
		}
#pragma omp atomic
		res[i] += sum;
	}
}


int* matrix_vector_omp_block(int** matrix, int* vector, int m, int n, double& dt)
{

	int* res = new int[m];
	for (int i = 0; i < m; ++i)
	{
		res[i] = 0;
	}
	double t1 = omp_get_wtime();
#pragma omp parallel 
	{
		block_vec_mult(matrix, vector, res, m, n, omp_get_thread_num());
	}
	double t2 = omp_get_wtime();
	dt = (t2 - t1) * 1000.0;
	return res;
}



int* matrix_vector(int** matrix, const int* vector, int m, int n, double& dt)
{
	int* res = new int[m];
	double t1 = omp_get_wtime();
	for (int i = 0; i < m; ++i)
	{
		int sum = 0;
		for (int j = 0; j < n; ++j)
		{
			sum += matrix[i][j] * vector[j];
		}
		res[i] = sum;
	}
	double t2 = omp_get_wtime();
	dt = (t2 - t1) * 1000.0;
	return res;
}


int** matrix_matrix_omp(int** mat1, int** mat2, int m, int n, int k, double& dt)
{
	int** res = new int* [m];
	for (int i = 0; i < m; ++i)
	{
		res[i] = new int[k];
	}
	double t1 = omp_get_wtime();
#pragma omp parallel for 
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			int sum = 0;
			for (int l = 0; l < n; ++l)
			{
				sum += mat1[i][l] * mat2[l][j];
			}
			res[i][j] = sum;
		}
	}
	double t2 = omp_get_wtime();
	dt = (t2 - t1) * 1000.0;
	return res;
}


void block_mat_mult(int** mat1, int** mat2, int** res, int n, int block_index)
{
	int block_size = n / 2;
	int block_row_start1 = block_index / 4;
	int block_col_start1 = block_index % 2;
	int block_col_start2 = block_index < 4 ? block_index / 2 : (block_index - 4) / 2;

	for (int i = block_row_start1 * block_size; i < (block_row_start1 + 1) * block_size; ++i)
	{
		for (int j = block_col_start2 * block_size; j < (block_col_start2 + 1) * block_size; ++j)
		{
			int sum = 0;
			for (int l = block_col_start1 * block_size; l < (block_col_start1 + 1) * block_size; ++l)
			{
				sum += mat1[i][l] * mat2[l][j];
			}
#pragma omp atomic
			res[i][j] += sum;
		}
	}
}


int** matrix_matrix_omp_block(int** mat1, int** mat2, int n, double& dt)
{
	int** res = new int* [n];
	for (int i = 0; i < n; ++i)
	{
		res[i] = new int[n];
		for (int j = 0; j < n; ++j)
		{
			res[i][j] = 0;
		}
	}
	const int block_num = 8; 
	double t1 = omp_get_wtime();
#pragma omp parallel for shared(res)
	for (int i = 0; i < block_num; ++i)
	{
		block_mat_mult(mat1, mat2, res, n, i);
	}
	double t2 = omp_get_wtime();
	dt = (t2 - t1) * 1000.0;
	return res;
}

int** matrix_matrix(int** mat1, int** mat2, int m, int n, int k, double& dt)
{
	int** res = new int* [m];
	for (int i = 0; i < m; ++i)
	{
		res[i] = new int[k];
	}
	double t1 = omp_get_wtime();
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			int sum = 0;
			for (int l = 0; l < n; ++l)
			{
				sum += mat1[i][l] * mat2[l][j];
			}
			res[i][j] = sum;
		}
	}
	double t2 = omp_get_wtime();
	dt = (t2 - t1) * 1000.0;
	return res;
}


int** generate_matrix(int m, int n)
{
	static int cnt;
	cnt++;
	int** res = new int* [m];
	srand(cnt);
	for (int i = 0; i < m; ++i)
	{
		res[i] = new int[n];
		for (int j = 0; j < n; ++j)
		{
			res[i][j] = rand() % 100;
		}
	}
	return res;
}

int* generate_vector(int n)
{
	static int cnt;
	cnt++;
	int* res = new int[n];
	srand(cnt);
	for (int i = 0; i < n; ++i)
	{
		res[i] = rand() % 100;
	}
	return res;
}

bool check(const int* vec1, const int* vec2, int n)
{
	for (int i = 0; i < n; ++i)
	{
		if (vec1[i] != vec2[i])
		{
			return false;
		}
	}
	return true;
}

bool check_matrix(int** mat1, int** mat2, int m, int n)
{
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			if (mat1[i][j] != mat2[i][j])
				return false;
		}
	}
	return true;
}


void delete_matrix(int** matrix, int m)
{
	for (int i = 0; i < m; ++i)
	{
		delete[] matrix[i];
	}
	delete[] matrix;
}


void matrix_vector_test(int m, int n, int count)
{
	int cnt = 0;
	double t_block_total = 0.0;
	double t_col_total = 0.0;
	double t_rows_total = 0.0;
	double t_total = 0.0;
	bool is_valid = true;
	while (cnt < count && is_valid)
	{
		double t_block, t_col, t_rows, t;
		int** matrix = generate_matrix(m, n);
		int* vec = generate_vector(n);
		int* res_omp_rows = matrix_vector_omp_rows(matrix, vec, m, n, t_rows);
		int* res_omp_col = matrix_vector_omp_columns(matrix, vec, m, n, t_col);
		int* res_omp_block = matrix_vector_omp_block(matrix, vec, m, n, t_block);
		int* res = matrix_vector(matrix, vec, m, n, t);

		t_block_total += t_block;
		t_col_total += t_col;
		t_rows_total += t_rows;
		t_total += t;

		if (!check(res, res_omp_rows, m))
		{
			printf("row result is false!!!!!!\n");
			is_valid = false;
		}
		if (!check(res_omp_block, res, m))
		{
			printf("block result is false!!!!!!\n");
			is_valid = false;
		}
		if (!check(res_omp_col, res, m))
		{
			printf("column result is false!!!!!!\n");
			is_valid = false;
		}
		++cnt;

		// free memory
		delete_matrix(matrix, m);
		delete[] vec;
		delete[] res;
		delete[] res_omp_col;
		delete[] res_omp_rows;
		delete[] res_omp_block;
	}
	if (is_valid)
	{
		printf("results of matrix-vector multiplication - average time in ms\n");
		printf("average time: %.2lf\n", t_total / count);
		printf("average omp time using rows: %.2lf\n", t_rows_total / count);
		printf("average omp time using columns: %.2lf\n", t_col_total / count);
		printf("average omp time using blocks: %.2lf\n", t_block_total / count);
	}
}


void matrix_matrix_test(int n, int count)
{
	int cnt = 0;
	double t_block_total = 0.0;
	double t_rows_total = 0.0;
	double t_total = 0.0;
	bool is_valid = true;
	while (cnt < count && is_valid)
	{
		double t_block, t_rows, t;
		int** mat1 = generate_matrix(n, n);
		int** mat2 = generate_matrix(n, n);
		int** mat_res_omp = matrix_matrix_omp(mat1, mat2, n, n, n, t_rows);
		int** mat_res = matrix_matrix(mat1, mat2, n, n, n, t);
		int** mat_omp_block = matrix_matrix_omp_block(mat1, mat2, n, t_block);

		if (!check_matrix(mat_res, mat_res_omp, n, n))
		{
			printf("row result of matrix mult is false!!!!\n");
			is_valid = false;
		}
		if (!check_matrix(mat_res, mat_omp_block, n, n))
		{
			printf("block result of matrix mult is false!!!!\n");
			is_valid = false;
		}

		t_block_total += t_block;
		t_rows_total += t_rows;
		t_total += t;

		++cnt;

		// free memory
		delete_matrix(mat_omp_block, n);
		delete_matrix(mat1, n);
		delete_matrix(mat2, n);
		delete_matrix(mat_res_omp, n);
		delete_matrix(mat_res, n);
	}
	if (is_valid)
	{
		printf("results of matrix-matrix multiplication - average time in ms\n");
		printf("average time: %.1lf\n", t_total / count);
		printf("average omp time using rows: %.1lf\n", t_rows_total / count);
		printf("average omp time using blocks: %.1lf\n", t_block_total / count);
	}

}


int main()
{
	int matrix_test_size, vector_test_size;
	const int count = 5;
	printf("Enter size for matrix-vector:");
	std::cin >> vector_test_size;
	printf("Enter size for matrix-matrix: ");
	std::cin >> matrix_test_size;
	omp_set_num_threads(num_threads);
	matrix_vector_test(vector_test_size, vector_test_size, count);
	matrix_matrix_test(matrix_test_size, count);

	return 0;
}