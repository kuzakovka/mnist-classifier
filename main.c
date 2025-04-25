#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"
#include <float.h>
#include <math.h>

int main() {
    // 1. Загрузка данных MNIST
    MnistRecord *records = malloc(MAX_RECORDS * sizeof(MnistRecord));
    if (!records) {
        perror("Memory allocation error");
        return 1;
    }

    int loaded = load_mnist("mnist_train.csv", records, MAX_RECORDS);
    if (loaded < 0) {
        free(records);
        return 1;
    }
    printf("Loaded %d records from mnist_train.csv\n", loaded);

    // 2. Загрузка конфигурации сети
    int *layer_sizes = NULL;
    int num_layers = 0;
    float learning_rate = 0.01f;  
    float regularization = 0.001f;
    
    // default network configuration
    if (!parse_config("config.txt", &layer_sizes, &num_layers, &learning_rate, &regularization)) {
        printf("Using default network configuration\n");
        int default_layers[] = {784, 256, 10};
        num_layers = 3;
        layer_sizes = malloc(num_layers * sizeof(int));
        for (int i = 0; i < num_layers; i++) {
            layer_sizes[i] = default_layers[i];
        }
    }

    // 3. Создание сети
    NeuralNetwork *net = create_network(layer_sizes, num_layers, learning_rate, regularization);
    if (!net) {
        free(layer_sizes);
        free(records);
        return 1;
    }

    printf("\nTraining network with configuration:\n");
    printf("Layers: ");
    for (int i = 0; i < num_layers; i++) {
        printf("%d ", layer_sizes[i]);
    }
    printf("\nLearning rate: %.4f, Regularization: %.4f\n\n", 
          learning_rate, regularization);
    
    

    // 4. Подготовка к обучению

    FILE* f = fopen("heatmap.txt", "w");
    if (f) fclose(f);  // Очистка файла для записи активаций

    int epochs = 45;
    float final_accuracy = 0.0f;
    printf("\nStarting training for %d epochs...\n", epochs);

    // 5. Цикл обучения
    for (int epoch = 0; epoch < epochs; epoch++) {
        int correct = 0;
        float epoch_loss = 0;

        for (int i = 0; i < loaded; i++) {
            // Выделяем память под градиенты (сумма размеров всех слоёв, кроме входного)
            int total_neurons = 0;
            for (int l = 1; l < net->num_layers; l++) {
                total_neurons += net->layers[l].size;
            }
            float *gradients = malloc(total_neurons * sizeof(float));

            // Обратное распространение
            backpropagation(net, records[i].pixels, records[i].label, gradients);
            
            // Кросс-энтропия каэдые 10 эпох
            if (epoch % 10 == 0) {
                float* output = forward_pass(net, records[i].pixels);
                epoch_loss += -logf(output[records[i].label] + FLT_EPSILON);
            }
        }
        // Кросс-энтропия каждые 10 эпох
        if (epoch % 10 == 0) {
            printf("Epoch %d: Average loss = %.4f\n", epoch, epoch_loss/60000.0f);   
        }
    }

    // 6. Сохранение результатов
    save_weights(net, "weights.bin");
    
    FILE *output = fopen("output.txt", "w");
    if (output) {
        fprintf(output, "Final accuracy: %.2f%%\n", final_accuracy * 100);
        fprintf(output, "Network architecture: ");
        for (int i = 0; i < net->num_layers; i++) {
            fprintf(output, "%d ", net->layers[i].size);
        }
        fprintf(output, "\nTraining epochs: %d\n", epochs);
        fclose(output);
        printf("Metrics saved to output.txt\n");
    }

        // ===== [Блок тестирования] =====
    MnistRecord *test_data = malloc(MAX_RECORDS * sizeof(MnistRecord));
    if (!test_data) {
        perror("Failed to allocate memory for test data");
        free_network(net);
        free(layer_sizes);
        free(records);
        return 1;
    }

    int test_loaded = load_mnist("mnist_test.csv", test_data, 9999);
    if (test_loaded < 0) {
        free(test_data);
        free_network(net);
        free(layer_sizes);
        free(records);
        return 1;
    }
    printf("\nLoaded %d TEST samples\n", test_loaded);

    // Проверка точности
    int correct = 0;
    for (int i = 0; i < test_loaded; i++) {
        float *output = forward_pass(net, test_data[i].pixels);
        
        // Находим предсказанный класс (индекс с максимальной вероятностью)
        int predicted = 0;
        for (int j = 1; j < 10; j++) {
            if (output[j] > output[predicted]) predicted = j;
        }

        if (predicted == test_data[i].label) correct++;

        // Вывод первых 3 примеров для наглядности
        if (i < 3) {
            printf("Sample %d: True = %d, Predicted = %d\n", 
                  i, test_data[i].label, predicted);
            printf("Probabilities: ");
            for (int j = 0; j < 10; j++) printf("%.3f ", output[j]);
            printf("\n---\n");
        }
    }

    float test_accuracy = (float)correct / test_loaded;
    printf("\nTest Accuracy: %.2f%% (%d/%d)\n", 
          test_accuracy * 100, correct, test_loaded);
    
    free(test_data);
    // ===== [Конец блока тестирования] =====

    // 7. Очистка
    free_network(net);
    free(layer_sizes);
    free(records);
    return 0;
}