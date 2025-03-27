#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define POP_SIZE 30
#define DIM 10  // Número de clientes
#define MAX_ITER 1000
#define BETA 3.0
#define ALPHA 0.5

// Função de custo (distância total percorrida) - é a função objetivo.
double custo(int *rota, double distancias[DIM][DIM]) {
    double total = 0.0;
    for (int i = 0; i < DIM - 1; i++) {
        total += distancias[rota[i]][rota[i + 1]];
    }
    total += distancias[rota[DIM - 1]][rota[0]]; // Retorno ao depósito
    return total;
}

// Inicialização das medusas (rotas aleatórias)
void inicializarPopulacao(int populacao[POP_SIZE][DIM]) {
    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < DIM; j++) {
            populacao[i][j] = j;
        }
        for (int j = 0; j < DIM; j++) {
            int swapIdx = rand() % DIM;
            int temp = populacao[i][j];
            populacao[i][j] = populacao[i][swapIdx];
            populacao[i][swapIdx] = temp;
        }
    }
}

// Perturbação para exploração (movimento Browniano)
void perturbar(int *rota) {
    int a = rand() % DIM;
    int b = rand() % DIM;
    int temp = rota[a];
    rota[a] = rota[b];
    rota[b] = temp;
}

// Algoritmo Jellyfish Search
void jellyfishSearch(double distancias[DIM][DIM]) {
    int populacao[POP_SIZE][DIM];
    inicializarPopulacao(populacao);
    
    int melhorRota[DIM];
    double melhorCusto = INFINITY;
    
    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < POP_SIZE; i++) {
            double custoAtual = custo(populacao[i], distancias); //utilização da função objetivo - voltada para achar o melhor custo.
            if (custoAtual < melhorCusto) {
                melhorCusto = custoAtual;
                for (int j = 0; j < DIM; j++) {
                    melhorRota[j] = populacao[i][j];
                }
            }
            perturbar(populacao[i]);
        }
    }
    
    printf("Melhor rota encontrada: ");
    for (int i = 0; i < DIM; i++) {
        printf("%d ", melhorRota[i]);
    }
    printf("\nCusto: %.2f\n", melhorCusto);
}

int main() {
    srand(time(NULL));
    double distancias[DIM][DIM];
    
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            distancias[i][j] = (i == j) ? 0.0 : (rand() % 100 + 1);
        }
    }
    
    clock_t inicio = clock();
    jellyfishSearch(distancias);
    clock_t fim = clock();
    
    double tempo_execucao = ((double)(fim - inicio)) / CLOCKS_PER_SEC;
    printf("Tempo de execucao: %.6f segundos\n", tempo_execucao);
}