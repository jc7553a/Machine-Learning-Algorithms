#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "utils.h"
double sigmoid(double x)
{
    return 1/(1 + exp(-x));
}


double sigmoid_derivative(double x)
{ 
	return sigmoid(x)*(1-sigmoid(x));
} 

void delta2_mult(double *a, double **w, double *b, double row, double column){
	for(int i = 0; i < column; i++){
		for (int j = 0; j < row; j++){
			b[i] = w[j][i]*a[j];
		}
	}
}

void vecs_to_matrix(double *a, double *b, double **m, int row, int column){
	for(int i = 0; i < row; i++){
		for (int j = 0; j < column; j++){
			m[i][j] = a[i]*b[j];
		}
	}
}

void matmul_vectors(double *a, double *b, double **m, int size_transpose, int size){
	for(int i = 0; i < size_transpose; i++){
		for (int j = 0; j < size; j++){
				m[j][i] = b[j]*a[i];
		}
	}
}

void scalar_multiply(double **m, double scalar, int row, int col){
	for (int i = 0; i < row; ++i){
		for (int j = 0; j < col; ++j){
			
			m[i][j] = m[i][j]*scalar;
		}
	}
}

void scalar_multiply_vector(double *vector, double scalar, int length){
	for(int i =0; i < length; ++i){
		vector[i] = vector[i]*scalar;
	}
}


void scalar_add_vector(double *vector, double scalar, int length){
	for(int i =0; i < length; ++i){
		vector[i] = vector[i]+scalar;
	}
}

void matrix_scalar_add(double **m, double scalar, double row, double col){
	for(int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			m[i][j] = m[i][j]*scalar;
		}
	}
}

void print_matrix(double **matrix, int row, int col){
	for(int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			printf("%.6f ", matrix[i][j]);
		}
		printf("\n");
	}
}

void matrix_element_add(double **w1, double **w2, int row, int column){
	for(int i = 0; i < row; i++){
		for (int j = 0; j < column; j++){
			w1[i][j] += w2[i][j];
			}
		}
}


void print_vector(double *v, int size){
    for(int i = 0; i < size; ++i){
        printf("%.6f ", v[i]);
    }
    printf("\n");
}