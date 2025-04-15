#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <cmath>
#include <mkl_cblas.h>
#include <iomanip>         // Подключаем для управления форматированием вывода

using namespace std;
using namespace std::chrono;

const int N = 256; // размерность матриц
using Complex = complex<float>; // Создаём псевдоним для комплекса

double compute_mflops(double time_seconds) {
    double c = 2.0 * N * N * N;  // Количество операций умножения и сложения для перемножения двух матриц
    return (c / time_seconds) * 1e-6;  // Производительность в MFLOPS
}

void static GenerateMatrix(vector<Complex>& mat) {
    random_device rd;                                   //
    mt19937 gen(rd());                                  //Создание RNG
    uniform_real_distribution<float> dis(-5.0f, 5.0f);  //

    //заполнение матрицы случайными числами
    for (auto& val : mat) {
        val = Complex(dis(gen), dis(gen));
    }
}

void print_result(const string& label, double time_s, double mflops) {
    cout << label << ":\n";
    cout << fixed << setprecision(6);  // Фиксированный формат с 6 знаками после запятой
    cout << "  Время: " << time_s << " сек\n";
    cout << "  Производительность: " << mflops << " MFLOPS\n\n";
}


void static multiply_naive(const vector<Complex>& A, const vector<Complex>& B, vector<Complex>& C) {
    for (int i = 0; i < N; ++i)  // Проходим по строкам матрицы A
       
        for (int k = 0; k < N; ++k) {  // Проходим по столбцам матрицы A
            Complex aik = A[i * N + k];  // Получаем элемент A(i,k)
            for (int j = 0; j < N; ++j)  // Проходим по столбцам матрицы B
                C[i * N + j] += aik * B[k * N + j];  // Выполняем умножение и добавляем в результат C
        }
}

// Оптимизированный вариант перемножения (блочный алгоритм)
void multiply_blocked(const vector<Complex>& A, const vector<Complex>& B, vector<Complex>& C, int block_size = 64) 
{
    // Блочная разбивка для повышения кэш-эффективности
    for (int ii = 0; ii < N; ii += block_size)
        for (int jj = 0; jj < N; jj += block_size)
            for (int kk = 0; kk < N; kk += block_size)
                for (int i = ii; i < min(ii + block_size, N); ++i)
                    for (int k = kk; k < min(kk + block_size, N); ++k) {
                       
                        Complex aik = A[i * N + k];  // Получаем элемент A(i,k)
                        for (int j = jj; j < min(jj + block_size, N); ++j)
                            C[i * N + j] += aik * B[k * N + j];  // Выполняем умножение и добавляем в результат C
                    }
}



int main()
{
    setlocale(LC_ALL, "rus");
    vector<Complex> A(N * N), B(N * N), C1(N * N), C2(N * N), C3(N * N);
    GenerateMatrix(A);
    GenerateMatrix(B);
 
   //1. Линейная алгебра
    auto start = high_resolution_clock::now();  // Засекаем время начала операции
    multiply_naive(A, B, C1);  // Перемножаем матрицы A и B с использованием обычного алгоритма
    auto end = high_resolution_clock::now();  // Засекаем время окончания операции
    double time_naive = duration<double>(end - start).count();  // Рассчитываем время работы
    print_result("1. Линейная алгебра", time_naive, compute_mflops(time_naive));  // Выводим результаты
   

    // 2. Использование BLAS (cblas_cgemm)
     start = high_resolution_clock::now();  // Засекаем время начала операции
    CBLAS_LAYOUT layout = CblasRowMajor;  // Устанавливаем формат хранения матриц по строкам
    CBLAS_TRANSPOSE trans = CblasNoTrans;  // Не транспонируем матрицы
    MKL_Complex8 alpha = { 1.0f, 0.0f };  // Коэффициент для первой матрицы
    MKL_Complex8 beta = { 0.0f, 0.0f };   // Коэффициент для второй матрицы
    int lda = N;
    int ldb = N;
    int ldc = N;
    cblas_cgemm(layout, trans, trans, N, N, N, &alpha,  // Выполняем умножение с помощью BLAS
        reinterpret_cast<const void*>(A.data()), lda,
        reinterpret_cast<const void*>(B.data()), ldb,
        &beta,
        reinterpret_cast<void*>(C2.data()), ldc);
     end = high_resolution_clock::now();  // Засекаем время окончания операции
    double time_blas = duration<double>(end - start).count();  // Рассчитываем время работы
    print_result("2. BLAS (MKL cblas_cgemm)", time_blas, compute_mflops(time_blas));  // Выводим результаты


    // 3. Оптимизированный вручную (блочный алгоритм)
    start = high_resolution_clock::now();  // Засекаем время начала операции
    multiply_blocked(A, B, C3);  // Перемножаем матрицы с использованием блочного алгоритма
    end = high_resolution_clock::now();  // Засекаем время окончания операции
    double time_blocked = duration<double>(end - start).count();  // Рассчитываем время работы
    print_result("3. Блочное перемножение", time_blocked, compute_mflops(time_blocked));  // Выводим результаты

    return 0;  // Завершаем программу
}

