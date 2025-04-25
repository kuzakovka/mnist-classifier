#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mnist.h"
#include <math.h>
#include <time.h>
#include <float.h> // Для DBL_EPSILON (малое число для защиты от переполнения)

int split_line(char *line, char *fields[], char delimiter) {
    int count = 0;
    char delimiters[2] = {delimiter, '\0'};
    char *token = strtok(line, delimiters);
    
    while (token != NULL && count < MAX_FIELDS) {
        fields[count++] = token;
        token = strtok(NULL, delimiters);
    }
    return count;
}

int load_mnist(const char *filename, MnistRecord *records, int max_records) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Ошибка открытия файла");
        return -1;
    }

    char line[MAX_LINE_LENGTH];
    char *fields[MAX_FIELDS];
    int record_count = 0;

    fgets(line, MAX_LINE_LENGTH, file); // Пропускаем заголовок

    while (fgets(line, MAX_LINE_LENGTH, file) && record_count < max_records) {
        line[strcspn(line, "\n")] = 0;
        
        int field_count = split_line(line, fields, ',');
        if (field_count != MAX_FIELDS) {
            fprintf(stderr, "Ошибка: неверное число полей в строке %d\n", record_count + 1);
            continue;
        }

        records[record_count].label = atoi(fields[0]);
        for (int i = 1; i < MAX_FIELDS; i++) {
            records[record_count].pixels[i-1] = atof(fields[i]) / 255.0f;
        }
        record_count++;
    }

    fclose(file);
    return record_count;
}

void print_mnist_record(const MnistRecord *record, int num_pixels) {
    printf("Label: %d\n", record->label);
    printf("First %d pixels: ", num_pixels);
    for (int i = 0; i < num_pixels; i++) {
        printf("%.2f ", record->pixels[i]);
    }
    printf("\n");
}


// Функция: считает количество чисел (слоёв) в строке "neurons: ..."
int count_numbers_in_line(const char *line) {
    int count = 1; // минимум 1 число
    for (int i = 0; line[i] != '\0'; i++) {
        if (line[i] == ',') {
            count++; // каждую запятую считаем как разделитель нового числа
        }
    }
    return count;
}

// Функция: читает конфигурацию сети из файла
int parse_config(
    const char *filename,
    int **layers,            // указатель на массив слоёв 
    int *num_layers,         // сколько всего слоёв
    float *learning_rate,    // скорость обучения
    float *regularization    // коэффициент регуляризации
) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Ошибка: не удалось открыть %s\n", filename);
        return 0;
    }

    char line[1024]; // строка для чтения каждой строки файла

    while (fgets(line, sizeof(line), file)) {
        // убираем символ переноса строки
        line[strcspn(line, "\n")] = 0;

        // если строка начинается с "neurons:"
        if (strncmp(line, "neurons:", 8) == 0) {
            char *start = strchr(line, ':'); // ищем двоеточие
            if (!start) continue;
            start++; // переходим к числам после двоеточия

            // сначала узнаем, сколько чисел (слоёв)
            *num_layers = count_numbers_in_line(start);

            // выделяем память под массив слоёв
            *layers = malloc(*num_layers * sizeof(int));

            // теперь парсим каждое число
            int i = 0;
            char *token = strtok(start, ", ");
            while (token && i < *num_layers) {
                (*layers)[i++] = atoi(token); // превращаем текст в число
                token = strtok(NULL, ", ");
            }
        }

        // если строка начинается с "learning_rate:"
        else if (strncmp(line, "learning_rate:", 14) == 0) {
            *learning_rate = atof(line + 14); // читаем число после двоеточия
        }

        // если строка начинается с "regularization:"
        else if (strncmp(line, "regularization:", 15) == 0) {
            *regularization = atof(line + 15); // читаем число после двоеточия
        }
    }

    fclose(file);
    return 1; 
}



// Создание нейросети
NeuralNetwork* create_network(const int *layers, int num_layers, float learning_rate, float regularization) {
    NeuralNetwork *net = malloc(sizeof(NeuralNetwork));
    net->num_layers = num_layers;
    net->learning_rate = learning_rate;
    net->regularization = regularization;
    net->layers = malloc(num_layers * sizeof(Layer));

    srand(time(NULL));
    
    for (int i = 0; i < num_layers; i++) {
        net->layers[i].size = layers[i];
        net->layers[i].output = malloc(layers[i] * sizeof(float));
        
        if (i > 0) {
            int prev_size = layers[i-1];
            net->layers[i].weights = malloc(prev_size * layers[i] * sizeof(float));
            net->layers[i].biases = malloc(layers[i] * sizeof(float));
            

            for (int j = 0; j < prev_size * layers[i]; j++) {
                net->layers[i].weights[j] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
            }
            
            for (int j = 0; j < layers[i]; j++) {
                net->layers[i].biases[j] = 0.0f;
            }
        }
    }
    return net;
}

// Освобождение памяти
void free_network(NeuralNetwork *net) {
    for (int i = 0; i < net->num_layers; i++) {
        free(net->layers[i].output);
        if (i > 0) {
            free(net->layers[i].weights);
            free(net->layers[i].biases);
        }
    }
    free(net->layers);
    free(net);
}


void softmax(float* x, int size) {
    if (size == 0) return;

    // 1. Находим максимум для численной стабильности
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // 2. Вычисляем экспоненты и их сумму
    float sum_exp = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = expf(x[i] - max_val);  // Используем expf для float
        sum_exp += x[i];
    }

    // 3. Нормализуем (делим каждый элемент на сумму)
    for (int i = 0; i < size; ++i) {
        x[i] /= sum_exp;
    }
}

void add_noise(float *pixels, int size, float noise_level) {
    for (int i = 0; i < size; i++) {
        if ((rand() % 100) < 10) {  // 10% вероятность
            // Генерируем шум в [0, noise_level]
            float noise = (rand() / (float)RAND_MAX) * noise_level;
            
            // Добавляем шум, но не превышаем 1.0
            if (pixels[i] + noise <= 1.0f) {
                pixels[i] += noise;
            } else {
                pixels[i] = 1.0f;  // Альтернатива: pixels[i] = 1.0f - (noise * 0.5f);
            }
        }
    }
}

void save_activations(NeuralNetwork* net) {
    FILE* f = fopen("heatmap.txt", "a");
    for (int l = 1; l < net->num_layers; l++) {
        fprintf(f, "Layer %d: ", l);
        for (int n = 0; n < net->layers[l].size; n++) {
            fprintf(f, "%.4f ", net->layers[l].output[n]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "---\n");
    fclose(f);
}


float* forward_pass(NeuralNetwork *net, const float *input) {

    const float* effective_input = input;
    float noisy_input[784];

        // // Добавляем шум:
        // memcpy(noisy_input, input, 784 * sizeof(float));
        // add_noise(noisy_input, 784, 0.2f);
        // effective_input = noisy_input;
        // // Добавляем шум.

    // Копируем входные данные в первый слой
    for (int i = 0; i < net->layers[0].size; i++) {
        net->layers[0].output[i] = effective_input[i];
    }

    // Вычисляем выходы для каждого слоя
    for (int l = 1; l < net->num_layers-1; l++) {
        Layer *current = &net->layers[l];
        Layer *previous = &net->layers[l-1];
        
        for (int n = 0; n < current->size; n++) {
            // Вычисляем взвешенную сумму
            float sum = current->biases[n];
            
            for (int p = 0; p < previous->size; p++) {
                sum += previous->output[p] * current->weights[p * current->size + n];
            }
            
            current->output[n] = ReLU(sum);
        }
    }

    // последний слой
    int l = net->num_layers-1;
    Layer *current = &net->layers[l];
    Layer *previous = &net->layers[l-1];
    
    for (int n = 0; n < current->size; n++) {
        // Вычисляем взвешенную сумму
        float sum = current->biases[n];
        
        for (int p = 0; p < previous->size; p++) {
            sum += previous->output[p] * current->weights[p * current->size + n];
        }
        
        current->output[n] = sum;
    }
    softmax(net->layers[net->num_layers-1].output, net->layers[net->num_layers-1].size);

    // // Сохранение активаций
    // if (net->num_layers > 1) save_activations(net);

    // Возвращаем указатель на выходной слой
    return net->layers[net->num_layers-1].output;
}

void backpropagation(
    NeuralNetwork *net,
    const float *input,
    const int target,
    float *gradients  // Оригинальный указатель (не изменяется)
) {
    // 1. Прямой проход
    forward_pass(net, input);

    // 2. Градиент выходного слоя
    Layer *output_layer = &net->layers[net->num_layers - 1];
    for (int n = 0; n < output_layer->size; n++) {
        gradients[n] = output_layer->output[n] - (n == target ? 1.0f : 0.0f);
    }

    // 3. Обратное распространение (используем КОПИЮ указателя)
    float *current_grads = gradients;  // Начинаем с выходного слоя
    
    for (int l = net->num_layers - 2; l >= 1; l--) {
        Layer *current = &net->layers[l];
        Layer *next = &net->layers[l + 1];
        float *next_gradients = current_grads + next->size;

        for (int n = 0; n < current->size; n++) {
            float grad = 0.0f;
            
            for (int k = 0; k < next->size; k++) {
                grad += current_grads[k] * next->weights[n * next->size + k];
            }
            
            grad *= (current->output[n] > 0) ? 1.0f : 0.0f;
            next_gradients[n] = grad;
        }
        
        current_grads += next->size;  
    }

    // 4. Обновление весов (идём от выходного слоя к первому скрытому)
int grad_offset = 0;
// Начинаем с выходного слоя (grad_offset = 0)
for (int l = net->num_layers-1; l >= 1; l--) {
    Layer *current = &net->layers[l];
    Layer *prev = &net->layers[l-1];
    
    for (int n = 0; n < current->size; n++) {
        float grad = gradients[grad_offset + n];
        current->biases[n] -= net->learning_rate * grad;
        
        for (int p = 0; p < prev->size; p++) {
            float reg_term = net->regularization * current->weights[p * current->size + n];
            current->weights[p * current->size + n] -= net->learning_rate * 
                (grad * prev->output[p] + reg_term);
        }
    }
    
    // Сдвигаем offset только для скрытых слоёв
    if (l > 1) {
        grad_offset += current->size;
    }
}
}

// Функция для сохранения весов в бинарный файл
void save_weights(NeuralNetwork *net, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open weights file");
        return;
    }

    // Записываем структуру сети (количество слоёв и их размеры)
    fwrite(&net->num_layers, sizeof(int), 1, file);
    for (int i = 0; i < net->num_layers; i++) {
        fwrite(&net->layers[i].size, sizeof(int), 1, file);
    }

    // Записываем веса и смещения
    for (int l = 1; l < net->num_layers; l++) {
        Layer *current = &net->layers[l];
        int weights_count = net->layers[l-1].size * current->size;
        
        fwrite(current->weights, sizeof(float), weights_count, file);
        fwrite(current->biases, sizeof(float), current->size, file);
    }

    fclose(file);
    printf("Weights saved to %s\n", filename);
}
