#ifndef MNIST_H
#define MNIST_H

#define MAX_LINE_LENGTH 10000
#define MAX_FIELDS 785
#define MAX_RECORDS 60000

#define ReLU(x) ((x) > 0 ? (x) : 0)

/* Структура для хранения одной записи MNIST */
typedef struct {
    char label;                     // Метка класса (цифра 0-9)
    float pixels[MAX_FIELDS - 1];   // Нормализованные значения пикселей [0,1]
} MnistRecord;

/* Структура слоя нейросети */
typedef struct {
    int size;       // Количество нейронов
    float *weights; // Матрица весов
    float *biases;  // Вектор смещений
    float *output;  // Выходные активации
} Layer;

/* Структура нейронной сети */
typedef struct {
    Layer *layers;          // Массив слоёв
    int num_layers;         // Количество слоёв
    float learning_rate;    // Скорость обучения
    float regularization;   // Коэффициент L2-регуляризации
} NeuralNetwork;


/* Функции для работы с MNIST */

/**
 * Разбивает строку на токены по указанному разделителю.
 * @param line Указатель на входную строку.
 * @param fields Массив указателей для хранения токенов.
 * @param delimiter Символ-разделитель.
 * @return Количество найденных токенов или -1 в случае ошибки.
 */
int split_line(char *line, char *fields[], char delimiter);

/**
 * Загружает датасет MNIST из CSV-файла.
 * @param filename Имя файла с данными MNIST.
 * @param records Массив для хранения загруженных записей.
 * @param max_records Максимальное количество записей для загрузки.
 * @return Количество загруженных записей или -1 в случае ошибки.
 */
int load_mnist(const char *filename, MnistRecord *records, int max_records);

/**
 * Выводит информацию об одной записи MNIST в консоль.
 * @param record Указатель на запись MNIST.
 * @param num_pixels Количество пикселей для вывода.
 */
void print_mnist_record(const MnistRecord *record, int num_pixels);

/* Функции для работы с конфигом */

/**
 * Подсчитывает количество чисел в строке конфигурационного файла.
 * @param line Указатель на строку конфигурации.
 * @return Количество чисел в строке или -1 в случае ошибки.
 */
int count_numbers_in_line(const char *line);

/**
 * Парсит конфигурационный файл для настройки нейронной сети.
 * @param filename Имя конфигурационного файла.
 * @param layers Указатель на массив размеров слоёв (выделяется внутри функции).
 * @param num_layers Указатель на переменную для хранения количества слоёв.
 * @param learning_rate Указатель на переменную для хранения скорости обучения.
 * @param regularization Указатель на переменную для хранения коэффициента регуляризации.
 * @return 1 в случае успеха, 0 при ошибке.
 */
int parse_config(const char *filename, int **layers, int *num_layers,
                float *learning_rate, float *regularization);

/* Функции нейросети */

/**
 * Создаёт нейронную сеть с заданной архитектурой.
 * @param layers Массив размеров слоёв.
 * @param num_layers Количество слоёв.
 * @param learning_rate Скорость обучения.
 * @param regularization Коэффициент L2-регуляризации.
 * @return Указатель на созданную нейронную сеть или NULL при ошибке.
 */
NeuralNetwork* create_network(const int *layers, int num_layers,
                            float learning_rate, float regularization);

/**
 * Освобождает память, занятую нейронной сетью.
 * @param net Указатель на нейронную сеть.
 */
void free_network(NeuralNetwork *net);

/**
 * Добавляет случайный шум к входным данным.
 * @param pixels Массив значений пикселей.
 * @param size Размер массива пикселей.
 * @param noise_level Уровень шума (амплитуда).
 */
void add_noise(float *pixels, int size, float noise_level);

/**
 * Выполняет прямой проход через нейронную сеть.
 * @param net Указатель на нейронную сеть.
 * @param input Массив входных данных (пиксели).
 * @return Массив выходных активаций последнего слоя.
 */
float* forward_pass(NeuralNetwork *net, const float *input);

/**
 * Обучает нейронную сеть на датасете MNIST.
 * @param net Указатель на нейронную сеть.
 * @param data Массив записей MNIST.
 * @param num_samples Количество записей для обучения.
 * @param epochs Количество эпох обучения.
 */
void train_network(NeuralNetwork *net, MnistRecord *data, int num_samples, int epochs);

/**
 * Оценивает точность нейронной сети на тестовом датасете.
 * @param net Указатель на нейронную сеть.
 * @param data Массив тестовых записей MNIST.
 * @param num_samples Количество тестовых записей.
 * @return Доля правильно классифицированных примеров (точность).
 */
float evaluate_network(NeuralNetwork *net, MnistRecord *data, int num_samples);

/**
 * Выполняет обратное распространение ошибки для обновления весов.
 * @param net Указатель на нейронную сеть.
 * @param input Массив входных данных (пиксели).
 * @param target Целевая метка класса.
 * @param gradients Массив для хранения градиентов.
 */
void backpropagation(NeuralNetwork *net, const float *input,
                    const int target, float *gradients);

/* Сохранение весов */

/**
 * Сохраняет веса и смещения нейронной сети в бинарный файл.
 * @param net Указатель на нейронную сеть.
 * @param filename Имя файла для сохранения.
 */
void save_weights(NeuralNetwork *net, const char *filename);

#endif