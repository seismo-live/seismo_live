# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# <div style='background-image: url("../share/images/header.svg") ; padding: 0px ; background-size: cover ; border-radius: 5px ; height: 250px'>
#     <div style="float: right ; margin: 50px ; padding: 20px ; background: rgba(255 , 255 , 255 , 0.7) ; width: 50% ; height: 150px">
#         <div style="position: relative ; top: 50% ; transform: translatey(-50%)">
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">Scientific Python</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">A super quick crash course</div>
#         </div>
#     </div>
# </div>

# Seismo-Live: http://seismo-live.org
#
# ##### Authors:
# * Lion Krischer ([@krischer](https://github.com/krischer))
#
# ---

# This notebook is a very quick introduction to Python and in particular its scientific ecosystem in case you have never seen it before. It furthermore grants a possibility to get to know the [IPython/Jupyter notebook](http://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261). [See here for the official documentation](http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb) of the Jupyter notebook - a ton more information can be found online.
#
#
# A lot of motivational writing on *Why Python?* is out there so we will not repeat it here and just condense it to a single sentence: **Python is a good and easy to learn, open-source, general purpose programming language that happens to be very good for many scientific tasks (due to its vast scientific ecosystem).**
#
#
# #### Quick Reference on How to Use This Notebook
#
#
# <img src="images/notebook_toolbar.png" style="width:70%"></img>
#
# * `Shift + Enter`: Execute cell and jump to the next cell
# * `Ctrl/Cmd + Enter`: Execute cell and don't jump to the next cell
#
#
# #### Disclaimer
#
# The tutorials are employing Jupyter notebooks but these are only one way of using Python. Writing scripts to text files and executing them with the Python interpreter of course also works:
#
# ```bash
# $ python do_something.py
# ```
#
# Another alternative is interactive usage on the command line:
#
# ```bash
# $ ipython
# ```
#
# ## Notebook Setup
#
# First things first: In many notebooks you will find a cell similar to the following one. **Always execute it!** They do a couple of things:
# * Make plots appear in the browser (otherwise a window pops up)
# * Printing things works like this:
#
# ```python
# print("Hello")
# ```
#
# This essentially makes the notebooks work under Python 2 and Python 3.
#
# * Plots look quite a bit nicer (this is optional).
#

# +
# Plots now appear in the notebook.
# %matplotlib inline

import matplotlib.pyplot as plt
plt.style.use('ggplot')                            # Matplotlib style sheet - nicer plots!
plt.rcParams['figure.figsize'] = 12, 8             # Slightly bigger plots by default
# -

# ---
#
# ## Useful Links
#
# Here is collection of resources regarding the scientific Python ecosystem. They cover a number of different packages and topics; way more than we will manage today.
#
# If you have any question regarding some specific Python functionality you can consult the official [Python documenation](http://docs.python.org/).
#
# Furthermore a large number of Python tutorials, introductions, and books are available online. Here are some examples for those interested in learning more.
#
# * [Learn Python The Hard Way](http://learnpythonthehardway.org/book/)
# * [Dive Into Python](http://www.diveintopython.net/)
# * [The Official Python Tutorial](http://docs.python.org/2/tutorial/index.html)
# * [Think Python Book](http://www.greenteapress.com/thinkpython/thinkpython.html)
#
# Some people might be used to Matlab - this helps:
#
# * [NumPy for Matlab Users Introdution](http://wiki.scipy.org/NumPy_for_Matlab_Users)
# * [NumPy for Matlab Users Cheatsheet](http://mathesaurus.sourceforge.net/matlab-numpy.html)
#
#
# Additionally there is an abundance of resources introducing and teaching parts of the scientific Python ecosystem.
#
# * [NumPy Tutorial](http://wiki.scipy.org/Tentative_NumPy_Tutorial)
# * [Probabilistic Programming and Bayesian Methods for Hackers](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/): Great ebook introducing Bayesian methods from an understanding-first point of view with the examples done in Python.
# * [Python Scientific Lecture Notes](http://scipy-lectures.github.io/): Introduces the basics of scientific Python with lots of examples.
# * [Python for Signal Processing](http://python-for-signal-processing.blogspot.de/): Free blog which is the basis of a proper book written on the subject.
# * [Another NumPy Tutorial](http://www.loria.fr/~rougier/teaching/numpy/numpy.html), [Matplotlib Tutorial](http://www.loria.fr/~rougier/teaching/matplotlib/matplotlib.html)
#
# You might eventually have a need to create some custom plots. The quickest way to success is usually to start from some example that is somewhat similar to what you want to achieve and just modify it. These websites are good starting points:
#
# * [Matplotlib Gallery](http://matplotlib.org/gallery.html)
# * [ObsPy Gallery](http://docs.obspy.org/gallery.html)
# * [Basemap Gallery](http://matplotlib.org/basemap/users/examples.html)
#
#
# ---

# ## Core Python Crash Course
#
# This course is fairly non-interactive and serves to get you up to speed with Python assuming you have practical programming experience with at least one other language. Nonetheless please change things and play around an your own - it is the only way to really learn it!
#
# The first part will introduce you to the core Python language. This tutorial uses Python 3 but almost all things can be transferred to Python 2. If possible choose Python 3 for your own work!
#
#
# ### 1. Numbers
#
# Python is dynamically typed and assigning something to a variable will give it that type.

# +
# Three basic types of numbers
a = 1             # Integers
b = 2.0           # Floating Point Numbers
c = 3.0 + 4j      # Complex Numbers, note the use of j for the complex part


# Arithmetics work as expected.
# Upcasting from int -> float -> complex
d = a + b         # (int + float = float)
print(d)

e = c ** 2        # c to the second power, performs a complex multiplication
print(e)
# -

# ### 2. Strings

# Just enclose something in single or double quotes and it will become a string. On Python 3 it defaults to unicode strings, e.g. non Latin alphabets and other symbols.

# +
# You can use single or double quotes to create strings.
location = "New York"

# Concatenate strings with plus.
where_am_i = 'I am in ' + location

# Print things with the print() function.
print(location, 1, 2)
print(where_am_i)

# Strings have a lot of attached methods for common manipulations.
print(location.lower())

# Access single items with square bracket. Negative indices are from the back.
print(location[0], location[-1])

# Strings can also be sliced.
print(location[4:])
# -

# #### Exercise
#
# Save your name in all lower-case letters to a variable, and print a capitalized version of it. Protip: [Google for "How to capitalize a string in python"](http://www.google.com/search?q=how+to+capitalize+a+string+in+python). This works for almost any programming problem - someone will have had the same issue before!

# + {"tags": ["exercise"]}


# + {"tags": ["solution"]}
name = "lion"
print(name.capitalize())
# -

# ### 3. Lists

# Python has two main collection types: List and dictionaries. The former is just an ordered collection of objects and is introduced here.

# +
# List use square brackets and are simple ordered collections of things.
everything = [a, b, c, 1, 2, 3, "hello"]

# Access elements with the same slicing/indexing notation as strings.
# Note that Python indices are zero based!
print(everything[0])
print(everything[:3])
print(everything[2:-2])

# Negative indices are counted from the back of the list.
print(everything[-3:])

# Append things with the append method.
everything.append("you")
print(everything)
# -

# ### 4. Dictionaries
#
# The other main collection type in Python are dictionaries. They are similiar to associative arrays or (hash) maps in other languages. Each entry is a key-value pair.

# +
# Dictionaries have named fields and no inherent order. As is
# the case with lists, they can contain anything.
information = {
    "name": "Hans",
    "surname": "Mustermann",
    "age": 78,
    "kids": [1, 2, 3]
}

# Acccess items by using the key in square brackets.
print(information["kids"])

# Add new things by just assigning to a key.
print(information)
information["music"] = "jazz"
print(information)

# Delete things by using the del operator
del information["age"]
print(information)


# -

# ### 5. Functions
#
# The key to conquer a big problem is to divide it into many smaller ones and tackle them one by one. This is usually achieved by using functions.

# +
# Functions are defined using the def keyword.
def do_stuff(a, b):
    return a * b

# And called with the arguments in round brackets.
print(do_stuff(2, 3))

# Python function also can have optional arguments.
def do_more_stuff(a, b, power=1):
    return (a * b) ** power

print(do_more_stuff(2, 3))
print(do_more_stuff(2, 3, power=3))

# For more complex function it is oftentimes a good idea to
#explicitly name the arguments. This is easier to read and less error-prone.
print(do_more_stuff(a=2, b=3, power=3))
# -

# ### 6. Imports
#
# To use functions and objects not part of the default namespace, you have import them. You will have to do this a lot so it is necessary to learn how to do it.

# +
# Import anything, and use it with the dot accessor.
import math

a = math.cos(4 * math.pi)

# You can also selectively import things.
from math import pi

b = 3 * pi

# And even rename them if you don't like their name.
from math import cos as cosine
c = cosine(b)
# -

# How to know what is available?
#
# 1. Read the [documentation](https://docs.python.org/3/library/math.html)
# 2. Interactively query the module

print(dir(math))

# Typing the dot and the TAB will kick off tab-completion.

# math.

# In the IPython framework you can also use a question mark to view the documentation of modules and functions.

# math.cos?

# ### 7. Control Flow
#
# Loops and conditionals are needed for any non-trivial task. Please note that **whitespace matters in Python**. Everything that is indented at the same level is part of the same block. By far the most common loops in Python are for-each loops as shown in the following. While loops also exist but are rarely used.

# +
temp = ["a", "b", "c"]

# The typical Python loop is a for-each loop, e.g.
for item in temp:
    # Everything with the same indentation is part of the loop.
    new_item = item + " " + item
    print(new_item)

print("No more part of the loop.")
# -

# Useful to know is the range() function.
for i in range(5):
    print(i)

# The second crucial control flow structure are if/else conditional and they work the same as in any other language.

# +
# If/else works as expected.
age = 77

if age >= 0 and age < 10:
    print("Younger than ten.")
elif age >= 10:
    print("Older than ten.")
else:
    print("Wait what?")

# +
# List comprehensions are a nice way to write compact loops.
# Make sure you understand this as it is very common in Python.

a = list(range(10))
print(a)
b = [i for i in a if not i % 2]
print(b)

# Equivalant loop for b.
b = []
for i in a:
    if not i % 2:
        b.append(i)
print(b)


# -

# ### 8. Error Messages
#
# You will eventually run into some error messages. Learn to read them! The last line is often the one that matters - reading upwards traces the error back in time and shows what calls led to it. If stuck: just google the error message!

# +
def do_something(a, b):
    print(a + b + something_else)

# do_something(1, 2)
# -

# ## The Scientific Python Ecosystem
#
# The [SciPy Stack](https://www.scipy.org/stackspec.html) forms the basis for essentially all applications of scientific Python. Here we will quickly introduce the three core libraries:
#
# * `NumPy`
# * `SciPy`
# * `Matplotlib`
#
# The SciPy stack furthermore contains `pandas` (library for data analysis on tabular and time series data) and `sympy` (package for symbolic math), both very powerful packages, but we will omit them in this tutorial.

# ### 9. NumPy
#
# Large parts of the scientific Python ecosystem use NumPy, an array computation package offering N-dimensional, typed arrays and useful functions for linear algebra, Fourier transforms, random numbers, and other basic scientific tasks.

# +
import numpy as np

# Create a large array with with 1 million samples.
x = np.linspace(start=0, stop=100, num=1E6, dtype=np.float64)

# Most operations work per-element.
y = x ** 2

# Uses C and Fortran under the hood for speed.
print(y.sum())

# FFT and inverse
x = np.random.random(100)
large_X = np.fft.fft(x)
x = np.fft.ifft(large_X)
# -

# ### 10. SciPy
#
# `SciPy`, in contrast to `NumPy` which only offers basic numerical routines, contains a lot of additional functionality needed for scientific work. Examples are solvers for basic differential equations, numeric integration and optimization, spare matrices, interpolation routines, signal processing methods, and a lot of other things.

# +
from scipy.interpolate import interp1d

x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x ** 2 / 9.0)

# Cubic spline interpolation to new points.
f2 = interp1d(x, y, kind='cubic')(np.linspace(0, 10, num=101, endpoint=True))
# -

# ### 11. Matplotlib
#
# Plotting is done using `Matplotlib`, a package for greating high-quality static plots. It has an interface that mimics Matlab which many people are familiar with.

# +
import matplotlib.pyplot as plt

plt.plot(np.sin(np.linspace(0, 2 * np.pi, 2000)), color="green",
         label="Some Curve")
plt.legend()
plt.ylim(-1.1, 1.1)
plt.show()
# -

# ## Exercises
#
# #### Functions, NumPy, and Matplotlib
#
# A. Write a function that takes a NumPy array `x` and `a`, `b`, and `c` and returns
#
# $$
# f(x) = a x^2 + b x + c
# $$
#
# B. Plot the result of that function with matplotlib.

# + {"tags": ["exercise"]}


# + {"tags": ["solution"]}
import matplotlib.pyplot as plt
import numpy as np

def simple_poly(x, a, b, c):
    return a * x ** 2 + b * x + c

plt.plot(simple_poly(np.linspace(-5, 5), 10, 2, 2))
plt.show()
# -

# #### 99 Bottles of Beer
#
# *(stolen from http://www.ling.gu.se/~lager/python_exercises.html)*
#
#
# "99 Bottles of Beer" is a traditional song in the United States and Canada. It is popular to sing on long trips, as it has a very repetitive format which is easy to memorize, and can take a long time to sing. The song's simple lyrics are as follows:
#
# ```
# 99 bottles of beer on the wall, 99 bottles of beer.
# Take one down, pass it around, 98 bottles of beer on the wall.
# ```
#
# The same verse is repeated, each time with one fewer bottle. The song is completed when the singer or singers reach zero.
#
# Your task here is write a Python program capable of generating all the verses of the song.
#

# + {"tags": ["exercise"]}


# + {"tags": ["solution"]}
print("99 bottles of beer on the wall, 99 bottles of beer.")
for i in range(98, -1, -1):
    print("Take one down, pass it around, %i bottles of beer on the wall." % i)
# -

# #### Ceasar Cipher
#
# *(stolen from http://www.ling.gu.se/~lager/python_exercises.html)*
#
# In cryptography, a Caesar cipher is a very simple encryption techniques in which each letter in the plain text is replaced by a letter some fixed number of positions down the alphabet. For example, with a shift of 3, A would be replaced by D, B would become E, and so on. The method is named after Julius Caesar, who used it to communicate with his generals. ROT-13 ("rotate by 13 places") is a widely used example of a Caesar cipher where the shift is 13. In Python, the key for ROT-13 may be represented by means of the following dictionary:
#
# ```python
# key = {'a':'n', 'b':'o', 'c':'p', 'd':'q', 'e':'r', 'f':'s', 'g':'t', 'h':'u',
#        'i':'v', 'j':'w', 'k':'x', 'l':'y', 'm':'z', 'n':'a', 'o':'b', 'p':'c',
#        'q':'d', 'r':'e', 's':'f', 't':'g', 'u':'h', 'v':'i', 'w':'j', 'x':'k',
#        'y':'l', 'z':'m', 'A':'N', 'B':'O', 'C':'P', 'D':'Q', 'E':'R', 'F':'S',
#        'G':'T', 'H':'U', 'I':'V', 'J':'W', 'K':'X', 'L':'Y', 'M':'Z', 'N':'A',
#        'O':'B', 'P':'C', 'Q':'D', 'R':'E', 'S':'F', 'T':'G', 'U':'H', 'V':'I',
#        'W':'J', 'X':'K', 'Y':'L', 'Z':'M'}
# ```
#
# Your task in this exercise is to implement an decoder of ROT-13. Once you're done, you will be able to read the following secret message:
#
# ```
# Pnrfne pvcure? V zhpu cersre Pnrfne fnynq!
# ```
#
# **BONUS:** Write an encoder!

# + {"tags": ["exercise"]}


# + {"tags": ["solution"]}
sentence = "Pnrfne pvcure? V zhpu cersre Pnrfne fnynq!"

key = {'a':'n', 'b':'o', 'c':'p', 'd':'q', 'e':'r', 'f':'s', 'g':'t', 'h':'u',
       'i':'v', 'j':'w', 'k':'x', 'l':'y', 'm':'z', 'n':'a', 'o':'b', 'p':'c',
       'q':'d', 'r':'e', 's':'f', 't':'g', 'u':'h', 'v':'i', 'w':'j', 'x':'k',
       'y':'l', 'z':'m', 'A':'N', 'B':'O', 'C':'P', 'D':'Q', 'E':'R', 'F':'S',
       'G':'T', 'H':'U', 'I':'V', 'J':'W', 'K':'X', 'L':'Y', 'M':'Z', 'N':'A',
       'O':'B', 'P':'C', 'Q':'D', 'R':'E', 'S':'F', 'T':'G', 'U':'H', 'V':'I',
       'W':'J', 'X':'K', 'Y':'L', 'Z':'M'}

result = ""
for letter in sentence:
    if letter not in key:
        result += letter
    else:
        result += key[letter]
print(result)
