#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define POP_SIZE 100
#define DIM 200  // Número de clientes
#define MAX_ITER 1000
#define BETA 3.0
#define ALPHA 0.5

// Função de custo (distância total percorrida) - é a função objetivo.
double custo(int *rota, double distancias[DIM][DIM]) {
    double total = 0.0;
    int i;

    #pragma omp parallel for reduction(+:total) private(i)
    for (i = 0; i < DIM; i++) {
        int proximo = (i + 1) % DIM;
        total += distancias[rota[i]][rota[proximo]];
    }

    return total;
}


// Inicialização das medusas (rotas aleatórias)
void inicializarPopulacao(int populacao[POP_SIZE][DIM]) {
    #pragma omp parallel for
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
        double melhorCustoLocal = INFINITY;
        int melhorRotaLocal[DIM];

        #pragma omp parallel for
        for (int i = 0; i < POP_SIZE; i++) {
            double custoAtual = custo(populacao[i], distancias);
            if (custoAtual < melhorCustoLocal) {
                melhorCustoLocal = custoAtual;
                for (int j = 0; j < DIM; j++) {
                    melhorRotaLocal[j] = populacao[i][j];
                }
            }
            perturbar(populacao[i]);
        }

        #pragma omp critical
        {
            if (melhorCustoLocal < melhorCusto) {
                melhorCusto = melhorCustoLocal;
                for (int j = 0; j < DIM; j++) {
                    melhorRota[j] = melhorRotaLocal[j];
                }
            }
        }
    }

    printf("Melhor rota encontrada: ");
    for (int i = 0; i < DIM; i++) {
        printf("%d ", melhorRota[i]);
    }
    printf("\nCusto: %.2f\n", melhorCusto);
}


int main() {
    omp_set_num_threads(8); // Ajuste para o número ideal de threads
    printf("Numero de threads: %d\n", omp_get_max_threads());

    srand(time(NULL));
    double distancias[DIM][DIM];

#pragma omp parallel for collapse(2)
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            distancias[i][j] = (i == j) ? 0.0 : (rand() % 100 + 1);
        }
    }

    // Tempo sequencial
    omp_set_num_threads(1);
    double start_seq = omp_get_wtime();
    jellyfishSearch(distancias);
    double end_seq = omp_get_wtime();
    double T_seq = end_seq - start_seq;

    // Tempo paralelo
    omp_set_num_threads(8);
    double start_par = omp_get_wtime();
    jellyfishSearch(distancias);
    double end_par = omp_get_wtime();
    double T_par = end_par - start_par;

    // Cálculo do Speedup e Eficiência
    int num_threads = omp_get_max_threads();
    double speedup = T_seq / T_par;
    double eficiencia = speedup / num_threads;

    printf("\nTempo Sequencial: %f segundos\n", T_seq);
    printf("Tempo Paralelo: %f segundos\n", T_par);
    printf("Speedup: %f\n", speedup);
    printf("Eficiencia: %f\n", eficiencia);

    return 0;
}