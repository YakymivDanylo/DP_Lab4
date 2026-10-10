# Лабораторна робота 5. Розпізнавання рукописних цифр MNIST нейронними мережами (Keras)

Від перцептрона до глибокої згорткової мережі: callback-зупинка навчання, MLP, CNN з одним шаром згортки,
порівняння архітектур CNN, 5-кратна крос-валідація, ручний підбір гіперпараметрів, збереження найкращої моделі та
розпізнавання власних цифр. Усі експерименти виконано в Google Colab (Keras 3, TensorFlow 2.x, GPU T4).

## Ноутбуки

| № | Завдання | Colab |
|---|---|---|
| 1 | Callback: зупинка при 99% точності | [відкрити](https://colab.research.google.com/drive/1KGAj_Hg89o5E4is56dTjRCGrzgClkzbj?usp=sharing) |
| 2 | MLP на MNIST, розпізнавання прикладів і власних цифр | [відкрити](https://colab.research.google.com/drive/175Cse5QO5O_XwKIxWxYQjGAwdH1Qtkj2?usp=sharing) |
| 3 | Покращення MNIST: один Conv2D + один MaxPooling2D, поріг 99,5% | [відкрити](https://colab.research.google.com/drive/1J5pvA030_k-TEGgwNe3qtWu_Hi5mj1Ra?usp=sharing) |
| 4 | CNN_MNIST: архітектури, k-fold, гіперпараметри, фінальна модель | <ВСТАВТЕ ПОСИЛАННЯ НА Lab5_4_CNN_MNIST> |
| 5 | Розпізнавання зі збереженої моделі | <ВСТАВТЕ ПОСИЛАННЯ НА Lab5_5_recognize> |

## Результати

| Модель | Тест MNIST (10 000) | Помилка | Власні цифри (10) | Параметрів |
|---|---|---|---|---|
| Dense(512), завдання 1 (5 епох) | 97,59% | 2,41% | не перевірялась | 407 050 |
| MLP 784-784-10, 100 епох (завдання 2) | 98,59% | 1,41% | 7/10 (6→5, 7→2, 9→3) | 623 290 |
| MLP + Dropout 0,2 + EarlyStopping | ___% | ___% | 7/10 | 623 290 |
| Conv2D(32) + MaxPool + Dense(128) (завдання 3) | 98,40% | 1,60% | 8/10 (6→5, 8→3) | 693 962 |
| `cnn_simple` (Conv 5×5 + Dropout) | 98,83% | 1,17% | 8/10 (6→5, 7→2) | 592 074 |
| `cnn_medium` (30×5×5, 15×3×3) | 99,13% | 0,87% | 9/10 (6→5) | 59 933 |
| `cnn_deep` (2 блоки по 2 згортки) | 99,46% | 0,54% | 10/10 | 870 634 |
| **`cnn_deeper`** (3 блоки + BatchNorm) | **99,54%** | **0,46%** | **10/10** | 585 514 |
| `cnn_deeper`, перенавчена на повному train (9 епох) | 99,24% | 0,76% | 10/10 | 585 514 |

Результат MLP з покращеннями (`___`) візьміть із таблиці `res` у ноутбуці 2 (`results/lab5_2_results.csv`).

**Найкраща модель, `cnn_deeper`** (`models/cnn_deeper.keras`): три блоки `Conv2D 3×3 → BatchNorm → ReLU` (по дві згортки,
фільтри 32 → 64 → 128), після кожного блоку `MaxPooling2D 2×2` + `Dropout 0,3`; класифікатор `Dense(256)` + `Dropout 0,5` + `Dense(10, softmax)`.
Adam, categorical crossentropy, batch 200, EarlyStopping за `val_loss` із поверненням найкращих ваг, валідація 10%.

**Перенавчання на повному train.** Епоху для повторного навчання на всіх 60 000 зображень було вибрано за `val_loss`
(9 епох), що дало недонавчену модель (99,24% < 99,54%). Для розпізнавання використано `cnn_deeper`,
як модель із найменшою помилкою (п. 34 завдання); перенавчену версію збережено як `models/mnist_cnn_final.keras` для порівняння.

### 5-кратна крос-валідація (8 епох, без EarlyStopping)

| Модель | Середня точність | Std |
|---|---|---|
| `cnn_medium` | 98,86% | 0,13 |
| `cnn_deep` | 99,16% | 0,12 |
| `cnn_deeper` | 98,96% | 0,19 |

Порядок моделей за крос-валідацією відрізняється від порядку за тестом; різниці лежать у межах розкиду.

### Підбір гіперпараметрів (по одному параметру від базової конфігурації)

Базова: фільтри 32-64-128, ядро 3×3, Dropout 0,3, Dense 256, BatchNorm, batch 200, Adam. Відбір за `val_acc` (6 000 зображень).

| Експеримент | val_acc | Тест |
|---|---|---|
| RMSprop | 99,58% | 99,49% |
| без BatchNorm | 99,57% | 99,39% |
| Dense 128 | 99,47% | 99,40% |
| Dropout 0,2 | 99,47% | 99,42% |
| batch 128 | 99,43% | 99,49% |
| Dropout 0,4 | 99,42% | 99,33% |
| Dense 512 | 99,42% | 99,33% |
| базова | 99,40% | 99,24% |
| фільтри 64-128-256 (1,74 млн парам.) | 99,40% | 99,34% |
| ядро 5×5 | 99,40% | 99,43% |
| фільтри 16-32-64 (223 тис. парам.) | 99,37% | 99,28% |
| batch 256 | 99,37% | 99,40% |

Усі значення `val_acc` лежать у діапазоні 0,21 в.п., що близько до статистичної похибки (≈0,09 в.п. на 6 000 зображень):
суттєвого впливу окремих гіперпараметрів не виявлено. Базова конфігурація зупинилась на 7-й епосі (EarlyStopping за `val_loss`
повернув ваги 3-ї епохи), тому має найгірший тест. Повна таблиця: `results/hyperparameter_tuning.csv`, `results/lab5_results.csv`.

### Точність по класах та помилки (`cnn_deeper`-схема, фінальна матриця помилок: `plots/confusion_matrix.png`)

Найслабші класи: **9** (97,2%) і **6** (98,9%); найкращий: **1** (99,7%).
Найчастіші помилки: 9→5 (10), 9→4 (7), 6→5 (5), 9→7 (5), 5→3 (4), 9→8 (4).

### Розпізнавання

- **10 випадкових зображень датасету:** 10/10 (впевненість 100%), `results/recognize_dataset_10.csv`.
- **10 власних цифр:** 10/10 для `cnn_deeper`; найнижча впевненість у цифри 6 (88,9%, друге місце 5 з 11,1%).
  Детально: `results/recognize_own_10.csv`, `plots/recognize_own_10.png`.
- На власних цифрах прості CNN помиляються на 6 і 7, MLP на 6, 7 і 9 (`results/own_images_all_models.csv`).
  10/10 на десяти зображеннях не доводить 100% точності на довільному почерку (вибірка мала).

### Підготовка власних зображень

Зображення: темна цифра на білому фоні (PNG 1152×648). Функція `prepare`: прозорий фон → білий, відтінки сірого, **інверсія**
(у MNIST цифра біла на чорному), обрізка по контуру цифри, масштабування до 20×20 зі збереженням пропорцій, центрування
в полі 28×28, нормалізація /255, форма `(1, 28, 28, 1)`. Файли: `images/0.png … 9.png`.

## Як використати збережену модель

Потрібен Keras 3.

```python
import urllib.request, numpy as np, keras
from PIL import Image, ImageOps

BASE_URL = 'https://raw.githubusercontent.com/YakymivDanylo/DP_Labs/production/lab5/'
urllib.request.urlretrieve(BASE_URL + 'models/cnn_deeper.keras', 'cnn_deeper.keras')

model = keras.models.load_model('cnn_deeper.keras')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   # перекомпіляція

def prepare(path):
    im = Image.open(path).convert('RGBA')
    im = Image.alpha_composite(Image.new('RGBA', im.size, (255, 255, 255, 255)), im).convert('L')
    im = ImageOps.invert(im)                                   # MNIST: біла цифра на чорному
    ys, xs = np.where(np.asarray(im) > 30)
    im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    s = 20.0 / max(im.size)
    im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
    canvas = Image.new('L', (28, 28), 0)
    canvas.paste(im, ((28 - im.size[0]) // 2, (28 - im.size[1]) // 2))
    return (np.asarray(canvas, dtype='float32') / 255.0).reshape(1, 28, 28, 1)

p = model.predict(prepare('photo.png'), verbose=0)[0]
print(p.argmax(), f'{p.max()*100:.0f}%')
```

## Структура папки `lab5/`

```
README.md
notebooks/                 ноутбуки 1-5 (.ipynb)
models/
    cnn_deeper.keras       найкраща модель (99,54%)
    cnn_deep.keras, cnn_medium.keras, cnn_simple.keras
    mnist_cnn_final.keras  cnn_deeper, перенавчена на повному train (99,24%)
    mnist_dense.keras, mnist_dense_improved.keras   MLP із завдання 2
    mnist_conv_5_3.keras   модель із завдання 3
results/                   усі таблиці (lab5_results.csv, kfold_results.csv, hyperparameter_tuning.csv, ...)
plots/                     графіки навчання, матриця помилок, приклади розпізнавання
images/                    власні зображення 0.png ... 9.png
```

## Час виконання (Colab, GPU T4, орієнтовно)

Ноутбук 1: ≈ 1–2 хв; 2: ≈ 4–6 хв; 3: ≈ 2–3 хв; 4: ≈ 30–50 хв (з крос-валідацією та підбором гіперпараметрів); 5: ≈ 1–2 хв.
