#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define POP_SIZE 70
#define DIM 10  // Número de clientes
#define MAX_ITER 100
#define BETA 3.0
#define ALPHA 0.5

// Função de custo (distância total percorrida) - é a função objetivo.
double custo(int* rota, double distancias[DIM][DIM]) {
    double total = 0.0;
    for (int i = 0; i < DIM - 1; i++) {
        total += distancias[rota[i]][rota[i + 1]];
    }
    total += distancias[rota[DIM - 1]][rota[0]];
    return total;
}

// Inicializa população (rotas aleatórias)
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
void perturbar(int* rota) {
    int a = rand() % DIM;
    int b = rand() % DIM;
    int temp = rota[a];
    rota[a] = rota[b];
    rota[b] = temp;
}

// Algoritmo Jellyfish Search SEQUENCIAL
double jellyfishSearch_sequencial(double distancias[DIM][DIM], int* melhorRota) {
    int populacao[POP_SIZE][DIM];
    inicializarPopulacao(populacao);

    double melhorCusto = INFINITY;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < POP_SIZE; i++) {
            double custoAtual = custo(populacao[i], distancias);
            if (custoAtual < melhorCusto) {
                melhorCusto = custoAtual;
                for (int j = 0; j < DIM; j++) {
                    melhorRota[j] = populacao[i][j];
                }
            }
            perturbar(populacao[i]);
        }
    }
    return melhorCusto;
}

// Algoritmo Jellyfish Search com MPI (paralelo)
double jellyfishSearch_mpi(double distancias[DIM][DIM], int rank, int num_procs, int* melhorRotaLocal) {
    int populacao[POP_SIZE][DIM];
    inicializarPopulacao(populacao);

    double melhorCustoLocal = INFINITY;

    int inicio = (POP_SIZE / num_procs) * rank;
    int fim = (rank == num_procs - 1) ? POP_SIZE : (POP_SIZE / num_procs) * (rank + 1);

    double tempo_inicio_local = MPI_Wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = inicio; i < fim; i++) {
            double custoAtual = custo(populacao[i], distancias);
            if (custoAtual < melhorCustoLocal) {
                melhorCustoLocal = custoAtual;
                for (int j = 0; j < DIM; j++) {
                    melhorRotaLocal[j] = populacao[i][j];
                }
            }
            perturbar(populacao[i]);
        }
    }

    double tempo_fim_local = MPI_Wtime();
    double tempo_exec_local = tempo_fim_local - tempo_inicio_local;

    // Redução para encontrar a melhor solução global
    double melhorCustoGlobal;
    MPI_Reduce(&melhorCustoLocal, &melhorCustoGlobal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Bcast(&melhorCustoGlobal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return tempo_exec_local;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    srand(time(NULL) + rank);
    double distancias[DIM][DIM];

    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            distancias[i][j] = (i == j) ? 0.0 : (rand() % 100 + 1);
        }
    }

    int melhorRotaSeq[DIM];
    int melhorRotaPar[DIM];

    double tempo_inicio_seq, tempo_fim_seq, seq_time = 0.0;
    double melhorCustoSeq = 0.0;

    // Apenas rank 0 executa a versão SEQUENCIAL
    if (rank == 0) {
        tempo_inicio_seq = MPI_Wtime();
        melhorCustoSeq = jellyfishSearch_sequencial(distancias, melhorRotaSeq);
        tempo_fim_seq = MPI_Wtime();
        seq_time = tempo_fim_seq - tempo_inicio_seq;

        printf("==== EXECUCAO SEQUENCIAL ====\n");
        printf("Melhor custo sequencial: %.2f\n", melhorCustoSeq);
        printf("Tempo sequencial: %.6f segundos\n\n", seq_time);
    }

    // Todos aguardam antes de começar a execução paralela
    MPI_Barrier(MPI_COMM_WORLD);

    // Cada processo mede seu próprio tempo de execução
    double tempo_exec_local = jellyfishSearch_mpi(distancias, rank, num_procs, melhorRotaPar);

    // Todos os processos recebem o tempo sequencial para cálculo do speedup
    MPI_Bcast(&seq_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Cada processo calcula seu speedup local
    double speedup_local = seq_time / tempo_exec_local;
    printf("Rank %d -> Tempo local: %.6f segundos, Speedup: %.2f\n", rank, tempo_exec_local, speedup_local);

    // O rank 0 calcula o speedup global
    if (rank == 0) {
        printf("\n==== EXECUCAO PARALELA ====\n");
        printf("Tempo paralelo (media dos processos): %.6f segundos\n", tempo_exec_local);

        // Cálculo do speedup global
        double speedup_global = seq_time / tempo_exec_local;
        printf("\n==== SPEEDUP ====\n");
        printf("Speedup global: %.2f\n", speedup_global);
    }

    MPI_Finalize();
    return 0;
}
