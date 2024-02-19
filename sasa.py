from transformers import AutoTokenizer, AutoModelWithLMHead


tokenizer = AutoTokenizer.from_pretrained("Sentdex/GPyT")
model = AutoModelWithLMHead.from_pretrained("Sentdex/GPyT").to("cuda")


def generate(code, max_length=100):
    '''Takes input code, replaces newline chars with <N>, 
    tokenizes, feeds thru model, decodes, 
    then reformats the newlines back in'''
    newlinechar = "<N>"
    converted = code.replace("\n", newlinechar)
    tokenized = tokenizer.encode(converted, return_tensors='pt').to("cuda")
    resp = model.generate(tokenized, max_length=max_length).to("cuda")

    decoded = tokenizer.decode(resp[0])
    reformatted = decoded.replace("<N>","\n")
    return reformatted


print(generate("import"))


import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame, Series, date_range
import pandas._testing as tm





print(generate("import", max_length=500))





import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame, Series, date_range
import pandas._testing as tm


class TestDataFrameToDatetime:
    def test_to_json_multiindex(self):
        # GH#17043
        df = DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [1.0, 2.0, 3.0, 4.0],
                "c": [1.0, 2.0, 3.0, 4.0],
                "d": [1.0, 2.0, 3.0, 4.0],
                "e": [1.0, 2.0, 3.0, 4.0],
                "f": [1.0, 2.0, 3.0, 4.0],
                "g": [1.0, 2.0, 3.0, 4.0],
                "h": [1.0, 2.0, 3.0, 4.0],
                "i": [1.0, 2.0, 3.0, 4.0],
                "j": [1.0, 2.0, 3.0, 4.0],
                "k": [1.0, 2.0, 3.0, 4.0],
            }
        )

        expected = DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
            }
        )
        assert result == expected






inp = """import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [5, 6, 2]"""

print(generate(inp))



import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [5, 6, 2]  # [1, 2, 3]

plt.figure()
plt.plot(x, y)
plt.plot(x, y)
plt.plot(x, y)
plt.plot(x, y)
plt.plot(x, y)



import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [5, 6, 2]  # [1, 2, 3]

plt.figure()
plt.plot(x, y)
plt.plot(x, y)
plt.plot(x, y)
plt.plot(x, y)
plt.plot(x, y)



inp = """import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [5, 6, 2]

# scatterplot
"""

print(generate(inp))

import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [5, 6, 2]

# scatterplot
plt.scatter(x, y, c='r', label='x')
plt.scatter(x, y, c='r', label='y')
plt.scatter(x, y, c='r', label='x')
plt.scatter(x, y, c='r', label='x')


import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [5, 6, 2]

# scatterplot
plt.scatter(x, y, c='r', label='x')




inp = """import matplotlib.pyplot as plt

x = [1, 2, 3]
# y = [5, 6, 2]

# histogram
"""

print(generate(inp))


import matplotlib.pyplot as plt

x = [1, 2, 3]
# y = [5, 6, 2]

# histogram
plt.hist(x, bins=5)
plt.title('Histogram')
plt.ylabel('Histogram')
plt.legend()

plt.show()

plt.figure()

plt.plot(x, y)<N


import matplotlib.pyplot as plt

x = [1, 2, 3]
# y = [5, 6, 2]

# histogram
plt.hist(x, bins=5)
plt.title('Histogram')
plt.ylabel('Histogram')
plt.legend()





def next_line_only(original, model_out):
    orig_nl = original.count("\n")
    one_more_lines = [l for l in model_out.splitlines(True)][:orig_nl+1]
    one_more_line = "".join(one_more_lines)
    return one_more_line



inp = """# graphing:
import matplotlib

# web requests:
import requests

# array math:
"""

print(next_line_only(inp, generate(inp)))


# graphing:
import matplotlib

# web requests:
import requests

# array math:
import numpy as np



inp = """# graphing:
import matplotlib

# web requests:
import requests

# neural networks
"""

print(next_line_only(inp, generate(inp)))
    


# graphing:
import matplotlib

# web requests:
import requests

# neural networks
from keras.layers import Dense, Dropout, Flatten, Flatten



inp = """# graphing:
import matplotlib

# web requests:
import requests

# build website
"""

print(next_line_only(inp, generate(inp)))


# graphing:
import matplotlib

# web requests:
import requests

# build website
from flask import Flask, request, jsonify, request

inp = """from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
"""

#print(next_line_only(inp, generate(inp)))

print(generate(inp))

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html", methods=['GET'])

@app.route('/index')
def index():
    return render_template("index.html", methods=['GET'])

@app.route('/index')
def index():
    return render_

def stop_at_repeat(model_out):
    lines = model_out.splitlines(True)
    no_repeat = ""
    for l in lines:
        if no_repeat.count(l) == 0 or l == "\n":
            no_repeat += l
        else:
            return no_repeat
    return no_repeat



inp = """from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
"""

#print(next_line_only(inp, generate(inp)))

m = generate(inp)

print(stop_at_repeat(m))

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html", methods=['GET'])




"""import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

# Conv:
"""

m = generate(inp, max_length=500)
print(stop_at_repeat(m))

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

# Conv:
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
model.add(MaxPooling2D((2, 2), strides=(1, 1), padding="same"))

inp = """import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

# Conv:
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
model.add(MaxPooling2D((2, 2), strides=(1, 1), padding="same"))

# Flatten:
"""

m = generate(inp, max_length=500)
print(stop_at_repeat(m))


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

# Conv:
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
model.add(MaxPooling2D((2, 2), strides=(1, 1), padding="same"))

# Flatten:
model.add(Flatten())



inp = """import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

# Conv:
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
model.add(MaxPooling2D((2, 2), strides=(1, 1), padding="same"))

# Flatten:
model.add(Flatten())

# Output 10 classes:
"""

m = generate(inp, max_length=500)
print(stop_at_repeat(m))

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

# Conv:
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
model.add(MaxPooling2D((2, 2), strides=(1, 1), padding="same"))

# Flatten:
model.add(Flatten())

# Output 10 classes:
model.add(Dense(10))

inp = """class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU()
        )

    def forward(self, x):
"""

m = generate(inp, max_length=200)
print(m)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = self.linear_relu_stack(x)
        return x


class NeuralNetwork(nn.Module):
    def __init__(self):