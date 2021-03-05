#include <omp.h>
#include <iostream>


class Integral {
private:
	double x;
	double y;
	double (*integral)(double);
	int steps = 10000000;
public:
	Integral(int x, int y, double(*integral)(double)) : x(x), y(y), integral(integral){}
	void show() {
		printf("x: %lf,\ny: %lf,\nintegral(x = 0): %lf\n", this->x, this->y, this->integral(0.0));
	}
	double calculate() {
		double t1 = omp_get_wtime();
		double sum = 0.0;
		double x = this->x;
		double dx = y / steps;
		for (int i = 0; i < steps; ++i) {
			sum += 4.0 / (1.0 + x * x);
			x += dx;
		}
		double t2 = omp_get_wtime();
		printf("Calculate time: %lf\n", t2 - t1);
		return sum * dx;
	}

	double calculate_parallel_for() {
		double t1 = omp_get_wtime();
		double sum = 0.0;
		double dx = y / steps;
		#pragma omp parallel for
		for (int i = 0; i < steps; ++i) {
			double x = i * dx;
			sum += 4.0 / (1.0 + x * x);
		}
		double t2 = omp_get_wtime();
		printf("Calculate_parallel_for time: %lf\n", t2 - t1);
		return sum * dx;
	}

	double calculate_parallel_reduction() {
		double t1 = omp_get_wtime();
		double sum = 0.0;
		double dx = y / steps;
		#pragma omp parallel for reduction(+: sum)
		for (int i = 0; i < steps; ++i) {
			double x = i * dx;
			sum += 4.0 / (1.0 + x * x);
		}
		double t2 = omp_get_wtime();
		printf("Calculate_parallel_reduction time: %lf\n", t2 - t1);
		return sum * dx;
	}

};

double integral(double x) {
	return (4.0 / (1.0 + x * x));
}

int main() {
	Integral *pi = new Integral(0.0, 1.0, integral);
	std::cout << pi->calculate() << std::endl;
	std::cout << pi->calculate_parallel_for() << std::endl;
	std::cout << pi->calculate_parallel_reduction() << std::endl;
	return 0;
}