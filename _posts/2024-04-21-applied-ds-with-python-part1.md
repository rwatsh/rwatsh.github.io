---
layout: post
title: Applied Data Science with Python - Part 1 - Python Refresher
date: 2024-04-21 08:11 -0500
tags: [books, ds]
categories: [DS, Book-ADWP]
---
# Applied Data Science with Python

https://www.coursera.org/learn/python-data-analysis

The course covers various data analysis techniques, such as cleaning and manipulating data, running statistical analyses, and using functions like groupby, merge, and pivot tables. These skills are valuable in many industries and can help you make data-driven decisions.

## Python Refresher

#### Datatypes 

```python
print(type("hello")) # <class 'str'>
print(type(None)) # <class 'NoneType'>
print(type(1)) # <class 'int'>
print(type(1.0)) # <class 'float'>

# Python functions with default arguments
def add(x, y, z = 1, kind = 'subtract'):
    if kind == 'add':
        return x+y+z
    else:
        return x-y-z

print(type(add)) # <class 'function'>

# dictionary
x = {'a': 1, 'b': 2}
print(type(x)) # <class 'dict'>

# iterate over dictionary
for key, value in x.items():
    print(key, value)
print(x['a']) #  # get value of key 'a'

print(x.get('c', 3)) # get value of key 'c' or return 3 if key 'c' does not exist
x['c']=3 # add key 'c' with value 3
print(x) # {'a': 1, 'b': 2, 'c': 3}

# list
y = [1, 2, 3]
print(type(y)) # <class 'list'>
y.append(4)
print(y) # 1, 2, 3, 4]

# tuple
z = (1, 2, 3)
print(type(z)) # <class 'tuple'>
print(z) # (1, 2, 3)

# unpacking tuple into variables
a, b, c = z
print(a, b, c)
```

#### List and String 

```python
print([1, 2] + [3, 4]) # concatenation
print([1, 2] * 3) # replication
print(1 in [1, 2, 3]) # membership
print([1, 2, 3][0]) # indexing
print([1, 2, 3][1:]) # slicing
print([1, 2, 3][::-1]) # slicing
x = 'This is a string'
print(x[0]) # indexing
print(x[1:]) # slicing
print(x[::-1]) # slicing to reverse the string
```

Output:

```
[1, 2, 3, 4]
[1, 2, 1, 2, 1, 2]
True
1
[2, 3]
[3, 2, 1]
T
his is a string
gnirts a si sihT
```



#### Looping

```python
for item in y: # iterate through list
    print(item)

for i, item in enumerate(y): # enumerate returns index and item
    print(i, item)

i = 0
while i < len(y): # while loop
    print(y[i])
    i += 1
```



#### String

```python
# string
firstname = 'Watsh'
lastname = 'Rajneesh'
print(firstname + ' ' + lastname) # concatenation
print(firstname * 3) # replication
print ('Watsh' in firstname) # membership
fullname = 'Watsh Ranjit Singh Rajneesh'
print(fullname.split(' ')[0]) # split string into list of words
print(fullname.split(' ')[-1])
```

Output:

```
Watsh Rajneesh
WatshWatshWatsh
True
Watsh
Rajneesh
```



#### Date and Time

```python
import datetime as dt
import time as tm

print(tm.time()) # current time in seconds since epoch
dtnow = dt.datetime.fromtimestamp(tm.time())
print(dtnow)
print(dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.minute, dtnow.second)

# timedelta - used to add or subtract time from a date
delta = dt.timedelta(days = 100)
print(delta)
today = dt.date.today()
print(today)
print(today - delta)
print(today > today - delta)
```

Output:

```
1713679451.155692
2024-04-21 01:04:11.155776
2024 4 21 1 4 11
100 days, 0:00:00
2024-04-21
2024-01-12
True
```



#### Object

```python
# Python Objects and map()
class Person:
    department = 'School of Information'

    def set_name(self, new_name):
        self.name = new_name

    def set_location(self, new_location):
        self.location = new_location

person = Person()
person.set_name('Watsh')
person.set_location('Austin')
print('{} lives in {} and works in the department {}'.format(person.name, person.location, person.department))
```

Output:

```
Watsh lives in Austin and works in the department School of Information
```



#### map()

```python
store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest = map(min, store1, store2) # returns iterator
print(cheapest)
for item in cheapest: # iterate through iterator
    print(item)


people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

def split_title_and_name(person):
    return person.split(' ')[0] + ' ' + person.split(' ')[-1]

list(map(split_title_and_name, people))

print(list(map(lambda person: person.split(' ')[0] + ' ' + person.split(' ')[-1], people)))
```

Output:

```
<map object at 0x103a79580>
9.0
11.0
12.34
2.01
['Dr. Brooks', 'Dr. Collins-Thompson', 'Dr. Vydiswaran', 'Dr. Romero']
```

#### Lambda functions

```python
my_function = lambda a, b, c : a + b + c
print(my_function(1, 2, 3))

# list iteration
my_list = []
for number in range(0, 1000):
    if number % 2 == 0:
        my_list.append(number)
print(my_list)

# list comprehension - more concise way to write the above code
my_list = [number for number in range(0, 1000) if number % 2 == 0]
my_list


def times_tables():
    lst = []
    for i in range(10):
        for j in range (10):
            lst.append(i*j)
    return lst

# alternative way to write the above function using list comprehension
times_tables() == [i*j for i in range(10) for j in range(10)]

lowercase = 'abcdefghijklmnopqrstuvwxyz'
digits = '0123456789'

answer = [a+b+c+d for a in lowercase for b in lowercase for c in digits for d in digits]
print(answer[:50])
```

Output:

```
6
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 9848, ...., 850, 852, 854, 856, 858, 860, 862, 864, 866, 868, 870, 872, 874, 876, 878, 880, 882, 884, 886, 888, 890, 892, 894, 896, 898, 900, 902, 904, 906, 908, 910, 912, 914, 916, 918, 920, 922, 924, 926, 928, 930, 932, 934, 936, 938, 940, 942, 944, 946, 948, 950, 952, 954, 956, 958, 960, 962, 964, 966, 968, 970, 972, 974, 976, 978, 980, 982, 984, 986, 988, 990, 992, 994, 996, 998]
['aa00', 'aa01', 'aa02', 'aa03', 'aa04', 'aa05', 'aa06', 'aa07', 'aa08', 'aa09', 'aa10', 'aa11', 'aa12', 'aa13', 'aa14', 'aa15', 'aa16', 'aa17', 'aa18', 'aa19', 'aa20', 'aa21', 'aa22', 'aa23', 'aa24', 'aa25', 'aa26', 'aa27', 'aa28', 'aa29', 'aa30', 'aa31', 'aa32', 'aa33', 'aa34', 'aa35', 'aa36', 'aa37', 'aa38', 'aa39', 'aa40', 'aa41', 'aa42', 'aa43', 'aa44', 'aa45', 'aa46', 'aa47', 'aa48', 'aa49']
```



