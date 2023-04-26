// Vladislav Grigorev
// 31.03.2023
// v.grigorev@innopolis.university
/*
 * This is a C++ program that defines several classes: Matrix, squareMatrix, identityMatrix, elimination and permutation Matrices.
 * The Matrix class has member functions to perform matrix addition, subtraction, and multiplication. 
 * Some classes are inherit from others. The task is to solve a system of linear equations A * x = b.
 * For this purpose we have class «ColumnVector» with necessary fields, methods and necessary operators' 
 * overloading for summation, multiplication, inputting-outputting and computing the norm.
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>

class permutationMatrix;

class eliminationMatrix;
class ColumnVector;

using namespace std;
int colNUM = 0;
int N;

/*
 * Matrix is a base class for matrices in general, with rows, columns, and matrix data members. It defines the basic
 * matrix operations of addition, subtraction, and multiplication, using overloaded operators. It also defines the
 * input and output stream operators to allow input and output of matrix objects.
 */
class Matrix {

public:
    vector<vector<double>> matrix;
    int rows{}, columns{};

    Matrix(double rows = 10, double cols = 10) : rows(rows), columns(cols) {
        matrix.resize(rows, vector<double>(cols));
    }

    Matrix operator+(Matrix &other);

    Matrix operator-(const Matrix &other);

    Matrix operator*(Matrix &other);


    friend ostream &operator<<(ostream &out, Matrix &mat) {
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.columns; j++) {
                if (round(mat.matrix[i][j] * 10000) / 10000 == 0) // handling with -0.00
                    cout << "0.0000";
                else
                    out << fixed << setprecision(4) << mat.matrix[i][j];
                if (j < mat.columns - 1) {
                    out << " ";
                }
            }
            out << endl;
        }
        return out;
    }

    friend istream &operator>>(istream &in, Matrix &mat) {
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.columns; j++) {
                in >> mat.matrix[i][j];
            }
        }
        return in;
    }


    void operator=(const Matrix &other) {
        rows = other.rows;
        columns = other.columns;
        matrix = other.matrix;
    }

    friend Matrix operator*(const Matrix &mat, const ColumnVector &other);
};

/*
 *This class called ColumnVector which represents a column vector b in Ax=b
 */
class ColumnVector {
public:
// Define properties
    vector<double> values; // Vector to hold values
    int rows{}, columns{}; // Number of rows and columns, columns is always 1

    // Define a constructor method that takes a vector of doubles as an argument
    ColumnVector(vector<double> values) {
        // Initialize properties based on the vector of doubles
        this->values = values;
        this->rows = values.size();
        this->columns = 1;
    }

    // Define overloaded operators
    ColumnVector operator+(const ColumnVector &other) const {
        // Add two column vectors together
        vector<double> resultValues(values.size());
        for (int i = 0; i < values.size(); i++) {
            resultValues[i] = values[i] + other.values[i];
        }
        return ColumnVector(resultValues); // Return the resulting vector as a new ColumnVector object
    }

    ColumnVector operator*(const double scalar) const {
        // Multiply the column vector by a scalar
        vector<double> resultValues(values.size());
        for (int i = 0; i < values.size(); i++) {
            resultValues[i] = values[i] * scalar;
        }
        return ColumnVector(resultValues); // Return the resulting vector as a new ColumnVector object
    }

    // Define a method to calculate the Euclidean norm of the column vector
    double norm() const {
        double sumOfSquares = 0;
        for (double value: values) {
            sumOfSquares += value * value;
        }
        return sqrt(sumOfSquares); // Return the square root of the sum of squares
    }

    // Define a friend function to allow the column vector to be printed to the console
    friend ostream &operator<<(ostream &output, const ColumnVector &vec) {
        for (double value: vec.values) {
            if (round(value * 100) / 100 == 0) // Check if the value is -0,00
                cout << "0.00" << endl; // If so, print 0.00
            else
                output << fixed << setprecision(2) << value
                       << endl; // Otherwise, print the value with two decimal places
        }
        return output;
    }

    // Define a friend function to read in column vector
    friend istream &operator>>(istream &in, ColumnVector &mat) {
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.columns; j++) {
                in >> mat.values[i];
            }
        }
        return in;
    }


    // Define an overloaded operator to allow the assignment of one column vector to another
    ColumnVector &operator=(const ColumnVector &other) {
        rows = other.rows;
        columns = other.columns;
        values = other.values;
        return *this; // Return the new vector
    }

    // Define friend functions to allow multiplication of permutation and elimination matrices by column vectors
    friend ColumnVector operator*(const permutationMatrix &matrix, const ColumnVector &other);

    friend ColumnVector operator*(const eliminationMatrix &matrix, const ColumnVector &other);
};

/*
 * several overloaded operators
 */
Matrix Matrix::operator+(Matrix &other) {
    if (rows != other.rows || columns != other.columns) {
        cout << "Error: the dimensional problem occurred" << endl;
        return Matrix(other);
    }

    Matrix result(rows, columns);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result.matrix[i][j] = matrix[i][j] + other.matrix[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix &other) {
    if (rows != other.rows || columns != other.columns) {
        cout << "Error: the dimensional problem occurred\n";
        return Matrix(other);
    }

    Matrix result(rows, columns);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result.matrix[i][j] = matrix[i][j] - other.matrix[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator*(Matrix &other) {
    if (columns != other.rows) {
        cout << "Error: the dimensional problem occurred\n";
        return Matrix(other);
    }

    Matrix result(rows, other.columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.columns; j++) {
            double sum = 0;
            for (int k = 0; k < columns; k++) {
                sum += matrix[i][k] * other.matrix[k][j];
            }
            if (round(sum * 100) / 100 == 0)
                sum = 0.00;
            result.matrix[i][j] = sum;
        }
    }
    return result;
}


/*squareMatrix is a derived class of Matrix that represents square matrices. It has the same operations as Matrix,
 * but with additional input and output stream operators to allow input and output of square matrix objects.
 */

class squareMatrix : public Matrix {

public:
    explicit squareMatrix(int rows = 10, int cols = 10) : Matrix(rows, cols) {}

    friend istream &operator>>(istream &in, squareMatrix &mat) {
        in >> mat.rows;
        mat.columns = mat.rows;
        mat.matrix.resize(mat.rows, vector<double>(mat.columns));
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.columns; j++) {
                in >> mat.matrix[i][j];
            }
        }
        return in;
    }

    friend ostream &operator<<(ostream &out, squareMatrix &mat) {
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.columns; j++) {
                if (round(mat.matrix[i][j] * 100) / 100 == 0)
                    cout << "0.00";
                else
                    out << mat.matrix[i][j];
                if (j < mat.columns - 1) {
                    out << " ";
                }
            }
            out << endl;
        }
        return out;
    }

    squareMatrix &operator=(const squareMatrix &other) {
        rows = other.rows;
        columns = other.columns;
        matrix = other.matrix;
        return *this;
    }

    squareMatrix operator+(squareMatrix &other) {
        Matrix tmp = Matrix::operator+(other);
        auto ans = (squareMatrix *) (&tmp);
        return *ans;
    }

    squareMatrix operator-(const squareMatrix &other) {
        Matrix tmp = Matrix::operator-(other);
        auto ans = (squareMatrix *) (&tmp);
        return *ans;
    }

    squareMatrix operator*(squareMatrix &other) {
        Matrix tmp = Matrix::operator*(other);
        auto ans = (squareMatrix *) (&tmp);
        return *ans;
    }

};

/*
 * identityMatrix is a derived class of squareMatrix that represents the identity matrix. It is initialized with the
 * number of rows and columns, and generates a matrix with ones on the diagonal and zeros elsewhere.
 */
class identityMatrix : public squareMatrix {
public:
    identityMatrix(int rows = 10, int cols = 10) : squareMatrix(rows, cols) {
        this->rows = rows;
        this->columns = cols;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (i == j)
                    this->matrix[i][j] = 1;
            }
        }
    }

// overloaded output
    friend ostream &operator<<(ostream &out, identityMatrix &mat) {
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.columns; j++) {
                if (round(mat.matrix[i][j] * 100) / 100 == 0)
                    cout << "0.00";
                else
                    out << mat.matrix[i][j];
                if (j < mat.columns - 1) {
                    out << " ";
                }
            }
            out << endl;
        }
        return out;
    }

    virtual identityMatrix &operator=(const Matrix &other) {
        rows = other.rows;
        columns = other.columns;
        matrix = other.matrix;
        return *this;
    }


};

/*
 * Elimination matrix makes zero under the pivot
 * by multiplying E*A
 */
class eliminationMatrix : public identityMatrix {
public:
    eliminationMatrix(int rows = 10, int cols = 10) : identityMatrix(rows, cols) {
    }

    void eliminate(Matrix &matr) { // elimination under main diagonal
        bool flag = false;
        for (int i = 0, j = 0; i < matr.rows; i++, j++) {
            for (int k = i + 1; !flag && k < matr.rows; k++) {
                if (matr.matrix[k][j] != 0) {
                    this->matrix[k][j] = -matr.matrix[k][j] / matr.matrix[i][j];
                    flag = true;
                    if (k == (matr.rows - 1))
                        colNUM++;
                }
            }
        }
    }

    void eliminateBack(Matrix &matr) {      // elimination above main diagonal
        bool flag = false;
        for (int i = matr.rows - 1, j = i; i >= 0; i--, j--) {
            for (int k = i - 1; !flag && k >= 0; k--) {
                if (round(matr.matrix[k][j] * 100) / 100 != 0) {
                    matrix[k][j] = -matr.matrix[k][j] / matr.matrix[i][j];
                    flag = true;
                    if (k == 0)
                        colNUM--;
                }
            }
        }
    }

    friend ostream &operator<<(ostream &out, eliminationMatrix &mat) {
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.columns; j++) {
                out << mat.matrix[i][j];
                if (j < mat.columns - 1) {
                    out << " ";
                }
            }
            out << endl;
        }
        return out;
    }

    eliminationMatrix &operator=(const Matrix &other) {
        rows = other.rows;
        columns = other.columns;
        matrix = other.matrix;
        return *this;
    }

// apply elimination to matrix
    friend eliminationMatrix operator*(const eliminationMatrix &matrix, const Matrix &other) {
        if (matrix.columns != other.rows) {
            cout << "Error: the dimensional problem occurred\n";
            return eliminationMatrix(matrix);
        }

        eliminationMatrix result(matrix.rows, other.columns);

        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < other.columns; j++) {
                double sum = 0;
                for (int k = 0; k < matrix.columns; k++) {
                    sum += matrix.matrix[i][k] * other.matrix[k][j];
                }
                if (round(sum * 100) / 100 == 0)
                    sum = 0.00;
                result.matrix[i][j] = sum;
            }
        }
        return result;
    }

// apply elimination to identity matrix
    friend eliminationMatrix operator*(const eliminationMatrix &matrix, const identityMatrix &other) {
        if (matrix.columns != other.rows) {
            cout << "Error: the dimensional problem occurred\n";
            return eliminationMatrix(matrix);
        }

        eliminationMatrix result(matrix.rows, other.columns);

        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < other.columns; j++) {
                double sum = 0;
                for (int k = 0; k < matrix.columns; k++) {
                    sum += matrix.matrix[i][k] * other.matrix[k][j];
                }
                if (round(sum * 100) / 100 == 0)
                    sum = 0.00;
                result.matrix[i][j] = sum;
            }
        }
        return result;
    }


};

/*
 * permutationMatrix inherits from identityMatrix
 * supporting permutation of rows
 */
class permutationMatrix : public identityMatrix {
public:
    permutationMatrix(int rows = 4, int cols = 4) : identityMatrix(rows, cols) {
    }

    void permutate(int a, int b) {
        matrix[a][a] = 0;
        matrix[b][a] = 1;
        matrix[b][b] = 0;
        matrix[a][b] = 1;
    }


    friend ostream &operator<<(ostream &out, permutationMatrix &mat) {
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.columns; j++) {
                out << mat.matrix[i][j];
                if (j < mat.columns - 1) {
                    out << " ";
                }
            }
            out << endl;
        }
        return out;
    }

    permutationMatrix &operator=(const permutationMatrix &other) {
        rows = other.rows;
        columns = other.columns;
        matrix = other.matrix;
        return *this;
    }

// apply permutation to matrix
    friend permutationMatrix operator*(const permutationMatrix &matrix, const Matrix &other) {
        if (matrix.columns != other.rows) {
            cout << "Error: the dimensional problem occurred\n";
            return permutationMatrix(matrix);
        }

        permutationMatrix result(matrix.rows, other.columns);

        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < other.columns; j++) {
                double sum = 0;
                for (int k = 0; k < matrix.columns; k++) {
                    sum += matrix.matrix[i][k] * other.matrix[k][j];
                }
                if (round(sum * 100) / 100 == 0)
                    sum = 0.00;
                result.matrix[i][j] = sum;
            }
        }
        return result;
    }

// apply permutation to the identity matrix
    friend permutationMatrix operator*(const permutationMatrix &matrix, const identityMatrix &other) {

        if (matrix.columns != other.rows) {
            cout << "Error: the dimensional problem occurred\n";
            return permutationMatrix(matrix);
        }

        permutationMatrix result(matrix.rows, other.columns);

        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < other.columns; j++) {
                double sum = 0;
                for (int k = 0; k < matrix.columns; k++) {
                    sum += matrix.matrix[i][k] * other.matrix[k][j];
                }
                if (round(sum * 100) / 100 == 0)
                    sum = 0.00;
                result.matrix[i][j] = sum;
            }
        }
        return result;
    }

};

// Define a friend function to allow multiplication of a permutation matrix and a column vector
ColumnVector operator*(const permutationMatrix &matrix, const ColumnVector &other) {
    if (matrix.columns != other.rows) {
        cout << "Error: the dimensional problem occurred\n";
        return ColumnVector(other);
    }
    ColumnVector result(vector<double>(other.rows));

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < other.columns; j++) {
            double sum = 0;
            for (int k = 0; k < matrix.columns; k++) {
                sum += matrix.matrix[i][k] * other.values[k];
            }
            if (round(sum * 100) / 100 == 0)
                sum = 0.00;
            result.values[i] = sum;
        }
    }
    return result;
}

// Define a friend function to allow multiplication of a elimination matrix and a column vector
ColumnVector operator*(const eliminationMatrix &matrix, const ColumnVector &other) {
    if (matrix.columns != other.rows) {
        cout << "Error: the dimensional problem occurred\n";
        return ColumnVector(other);
    }
    ColumnVector result(vector<double>(other.rows));

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < other.columns; j++) {
            double sum = 0;
            for (int k = 0; k < matrix.columns; k++) {
                sum += matrix.matrix[i][k] * other.values[k];
            }
            if (round(sum * 100) / 100 == 0)
                sum = 0.00;
            result.values[i] = sum;
        }
    }
    return result;
}

 Matrix operator*(const Matrix &mat, const ColumnVector &other) {
    if (mat.columns != other.rows) {
        cout << "Error: the dimensional problem occurred\n";
        return Matrix(mat);
    }

    Matrix result(mat.rows, other.columns);

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < other.columns; j++) {
            double sum = 0;
            for (int k = 0; k < mat.columns; k++) {
                sum += mat.matrix[i][k] * other.values[k];
            }
            if (round(sum * 10000) / 10000 == 0)
                sum = 0.0000;
            result.matrix[i][j] = sum;
        }
    }
    return result;
}


// end-when all zeroes under main diagonal
bool isAllZeroes(Matrix &matr) {
    for (int i = 1; i < matr.rows; i++) {
        for (int j = 0; j < i; j++) {
            if (round(matr.matrix[i][j] * 100) / 100 != 0)
                return false;
        }
    }
    return true;
}

// finds number of row when we have only zeroes
int isAllZeroesRow(Matrix &matr) {
    bool flag;
    for (int i = 0; i < matr.rows; i++) {
        flag = false;
        for (int j = 0; j < matr.columns; j++) {
            if (round(matr.matrix[i][j] * 100) / 100 == 0)
                flag = true;
            else {
                flag = false;
                break;
            }
        }
        if (flag)
            return i;
    }
    return -1;
}


// end-when all zeroes above main diagonal
bool isAllZeroesBack(Matrix &matr) {
    for (int i = matr.rows - 2; i >= 0; i--) {
        for (int j = matr.columns - 1; j > i; j--) {
            if (round(matr.matrix[i][j] * 100) / 100 != 0)
                return false;
        }
    }
    return true;
}

// find max by absolute value
bool isNeedPermutation(Matrix &matr) {
    for (int i = 0, j = 0; i < matr.rows; i++, j++) {
        bool flag = false;
        for (int k = i + 1; k < matr.rows; k++) {
//            cout << "$ "  << " " << k << " " << j << A.matrix[k][j]<< endl;
            if (matr.matrix[k][j] == 0)
                flag = true;
            else {
                flag = false;
                break;
            }
        }
        if (flag) {
//            cout << "J" << j << endl;
            colNUM = j + 1;
        } else
            break;
    }
    double pivot = matr.matrix[colNUM][colNUM];
    for (int k = colNUM + 1; k < matr.rows; k++) {
        if (abs(matr.matrix[k][colNUM]) > abs(pivot)) {
            return true;
        }
    }

    return false;
}

/*
 * this method creates permutation matrix and swaps rows
 */
void permutMatr(Matrix &A, identityMatrix &I) {
    double pivot = A.matrix[colNUM][colNUM];
    int remember = colNUM;

    for (int k = colNUM + 1; k < A.rows; k++) {
        if (abs(A.matrix[k][colNUM]) > abs(A.matrix[remember][colNUM])) {
            remember = k;
        }
    }
    auto P = new permutationMatrix(A.rows, A.columns);
    P->permutate(colNUM, remember);
    A = *P * A;
    I = *P * I;
    for (int i = 0, j = 0; i < A.rows; i++, j++) {
        bool flag = false;
        for (int k = i + 1; k < A.rows; k++) {
            if (A.matrix[k][j] == 0)
                flag = true;
            else {
                flag = false;
                break;
            }
        }
        if (flag) {
            colNUM = j + 1;
        } else
            break;
    }
}

int main() {
    colNUM = 0; // it's pivot
//    ios_base::sync_with_stdio(false);
//    cin.tie(nullptr);
    int m, n;
    cin >> m;
    vector<double> t(m);
    ColumnVector b((vector<double>(m)));
    for (int i = 0; i < m; i++) {
        cin >> t[i] >> b.values[i];
    }
    // Construct the matrix A
    cin >> n;
    Matrix A(m, n + 1);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n + 1; j++) {
            A.matrix[i][j] = pow(t[i], j);
        }
    }
    cout << "A:\n" << A;
    // Construct the transpose of A
    Matrix AT(n + 1, m);
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j < m; j++) {
            AT.matrix[i][j] = A.matrix[j][i];
        }
    }
    Matrix AT_A = AT * A;
    cout << "A_T*A:\n" << AT_A;

    identityMatrix I(AT_A.rows, AT_A.columns);
    if (!(isAllZeroes(AT_A))) // until we don't have all zeroes under main diagonal
        while (!(isAllZeroes(AT_A))) {
            if (isNeedPermutation(AT_A)) { // if we need swap rows(max by absolute value)
                permutMatr(AT_A, I); // create permutation matrix and multiply it by A
            } else {
                auto E = new eliminationMatrix(AT_A.rows, AT_A.columns);
                E->eliminate(AT_A); // if not we are creating elimination matrix
                AT_A = *E * AT_A;         // and multiply it by A
                I = (*E) * I;
            }
        }

    if (!(isAllZeroesBack(AT_A))) // until we don't have all zeroes above main diagonal
        while (!(isAllZeroesBack(AT_A))) {
            auto E = new eliminationMatrix(AT_A.rows, AT_A.columns);
            E->eliminateBack(AT_A); // make all zeroes above main diagonal
            AT_A = *E * AT_A;
            I = *E * I;
        }

    for (int i = 0, j = 0; i < AT_A.rows; i++, j++) { // doing diagonal normalization to obtain identity matrix A
        for (int k = 0; k < I.columns; k++) {
            I.matrix[i][k] /= AT_A.matrix[i][j];
        }
        AT_A.matrix[i][j] /= AT_A.matrix[i][j];
    }
    Matrix AT_Atr = I;
    cout << "(A_T*A)^-1:\n" << AT_Atr;
    Matrix AT_b = AT * b;
    cout << "A_T*b:\n" << AT_b;
    Matrix ans = (AT_Atr * AT_b);
    cout << "x~:\n" << ans;


}
