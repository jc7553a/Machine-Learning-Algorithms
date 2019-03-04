
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "utils.h"

int input_size = 3;
int hidden_size = 2;
double learning_rate = .05;
double b1 = .015;
double b2 = .015;

void matmul_inner(double **w, double *v, double *o, int row, int col);

void matmul_outer(double **w, double *v, double *o, int row, int col);


double compute_loss(double *input, double *output, int size)
{
    double loss = 0;
    for (int i =0; i < size; ++i){
        loss += pow((input[i] - output[i]),2);
    }
    return loss/(double)size;
}


void generate_variables(double *data, int size){
    for(int i = 0; i < size; ++i){
        data[i] = 1.0;
    }
}

void activate(double *hidden, int size){
    for(int i = 0; i < size; ++i){
        hidden[i] = sigmoid(hidden[i]);
    }
}

void backpropogate(double **w1, double **w2,  double *input, double *hidden, double *output, double *delta, double **adj_matrix2, double *hidden2, double *delta2, double **adj_matrix){
    double adj_bias2 = 0.0;
    double adj_bias1 = 0.0;
    for(int i =0; i < input_size; ++i){
        delta[i] = (output[i] - input[i])*output[i]*(1-output[i]);
        adj_bias2 -= learning_rate*delta[i]*b2;
    }
    
    matmul_vectors(hidden, delta, adj_matrix2, hidden_size, input_size);
    matrix_scalar_add(adj_matrix2, learning_rate*-1, input_size, hidden_size);
    for (int i =0; i < hidden_size; ++i){
        hidden2[i] = (hidden[i]*-1)+1;
    }

    delta2_mult(delta, w2, delta2, input_size, hidden_size);

    for (int i =0; i < hidden_size; ++i){
        delta2[i] = delta2[i]*hidden2[i]*hidden[i];
        adj_bias1 -=learning_rate*delta2[i]*b1;
    }
    vecs_to_matrix(hidden, input, adj_matrix, hidden_size, input_size);
    scalar_multiply(adj_matrix,learning_rate*-1, hidden_size, input_size);
    matrix_element_add(w1, adj_matrix, hidden_size, input_size);
    matrix_element_add(w2, adj_matrix2, input_size, hidden_size);
    b1 += adj_bias1;
    b2 += adj_bias2;

}


void feedforward(double **w1, double **w2, double *input, double *output, double *hidden){
    matmul_inner(w1, input, hidden, hidden_size, input_size);
    scalar_add_vector(hidden, b1, hidden_size);
    activate(hidden, hidden_size);
    matmul_outer(w2, hidden, output, input_size, hidden_size);
    scalar_add_vector(output, b2, input_size);
    activate(output, input_size);
}


void fill_weights(double **w, int row, int col){
    for (int i = 0; i < row; ++i){
        for (int j = 0; j < col; ++j){
           w[i][j] = (rand()/((double)RAND_MAX));
        }
    }
}



void matmul_inner(double **w, double *v, double *o, int row, int col){
    double count;
    for (int i = 0; i < row; ++i){
        count = 0;
        for (int j =0; j < col; ++j){
            count += w[i][j]*v[i];
        }
        o[i] = count;
    }
}

void matmul_outer(double **w, double *v, double *o, int row, int col){
    double count;
    for (int i = 0; i < row; ++i){
        count = 0;
        for (int j =0; j < col; ++j){
            count += w[i][j]*v[j];
        }
        o[i] = count;
    }
}



int main()
{
    int  i, j;
    double *input = (double *)malloc(input_size*sizeof(double));
    double *hidden = (double *)malloc(hidden_size*sizeof(double));
    double *hidden2 = (double *)malloc(hidden_size*sizeof(double));
    double *output = (double *)malloc(hidden_size*sizeof(double));
    double *delta = (double *)malloc(input_size*sizeof(double));
    double *delta2 = (double *)malloc(hidden_size*sizeof(double));
    double **adj_matrix2 = (double **)malloc(input_size*sizeof(double *));
    for(int i = 0; i<input_size; ++i){
        adj_matrix2[i] = (double *)malloc(hidden_size*sizeof(double *));
    }
    double **adj_matrix = (double **)malloc(hidden_size*sizeof(double *));
    for(int i = 0; i<hidden_size; ++i){
        adj_matrix[i] = (double *)malloc(input_size*sizeof(double *));
    }

    double loss;
    double **w1=(double **)malloc(hidden_size*sizeof(double *));
    for(i=0;i<hidden_size;i++){
        w1[i]=(double *)malloc(input_size*sizeof(double));
    }
    double **w2 = (double **)malloc(input_size*sizeof(double *));

    for(i = 0; i<input_size; ++i){
        w2[i] = (double *)malloc(hidden_size*sizeof(double *));
    }
    fill_weights(w1, hidden_size, input_size);
    fill_weights(w2, input_size, hidden_size);

    for (int i = 0; i < input_size; ++i){
        input[i] = .5;
    }

    for(int i = 0; i < 10000; ++i){
        feedforward(w1, w2, input, output, hidden);
        loss = compute_loss(input, output, input_size);
        if(i%1000 == 0){
            printf("Input \n");
            print_vector(input, input_size);
            printf("\nOutput \n");
            print_vector(output, input_size);
            printf("Loss : %.5f \n", loss);
        }
        
        backpropogate(w1, w2, input, hidden, output, delta, adj_matrix2, hidden2, delta2, adj_matrix);
    }
    return 0;
}

