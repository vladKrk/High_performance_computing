#include <omp.h>
#include <iostream>
#include <iomanip>
#include <algorithm>

class Matrix {
private:
	int n;
	int m;
	int** matrix;
public:
	Matrix(int n, int m): n(n), m(m) {
		matrix = new int* [n];
		std::cout << "  ";
		for (int i = 0; i < 10; ++i) {
			std::cout << std::setw(4) << i << ' ';
		}
		std::cout << std::endl;
		for (int i = 0; i < 10; ++i) {
			std::cout << std::setw(4) << "-" << ' ';
		}
		std::cout << std::endl;
		for (int i = 0; i < n; ++i) {
			matrix[i] = new int[m];
			std::cout << i << "| ";
			for (int j = 0; j < m; ++j) {
				matrix[i][j] = rand() % 1000;
				std::cout << std::setw(4) << matrix[i][j] << ' ';
			}
			std::cout << std::endl;
		}
	}

	int findMaxAmongMinRaw() {
		int max = -1;
		double t1 = omp_get_wtime();
		for (int i = 0; i < n; ++i) {
			int min = matrix[i][0];

			#pragma omp parallel for reduction(min: min)
			for (int j = 1; j < m; ++j) {
				if (matrix[i][j] < min) {
					min = matrix[i][j];
				}
			}
			if (min > max) {
				max = min;
			}
		}
		double dt = omp_get_wtime() - t1;
		printf("findMaxAmongMinRaw parallel time: %lf\n", dt);
		return max;
	}

	int findMaxAmongMinRawSync() {
		int max = -1;
		double t1 = omp_get_wtime();
		for (int i = 0; i < n; ++i) {
			int min = matrix[i][0];

			for (int j = 1; j < m; ++j) {
				if (matrix[i][j] < min) {
					min = matrix[i][j];
				}
			}
			if (min > max) {
				max = min;
			}
		}
		double dt = omp_get_wtime() - t1;
		printf("findMaxAmongMinRaw sync time: %lf\n", dt);
		return max;
	}


	~Matrix() {
		for (int i = 0; i < n; ++i) {
			delete[] matrix[i];
		}
		delete[] matrix;
	}
};

int main() {
	srand(time(NULL));
	Matrix *a = new Matrix(10, 10);
	int max = a->findMaxAmongMinRaw();
	std::cout << "Max: "<< max << std::endl;
	max = a->findMaxAmongMinRawSync();
	std::cout << "Max sync: " << max << std::endl;
	return 0;
}