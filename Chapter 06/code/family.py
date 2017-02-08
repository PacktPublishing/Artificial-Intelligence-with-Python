import json
from logpy import Relation, facts, run, conde, var, eq

# Check if 'x' is the parent of 'y'
def parent(x, y):
    return conde([father(x, y)], [mother(x, y)])

# Check if 'x' is the grandparent of 'y'
def grandparent(x, y):
    temp = var()
    return conde((parent(x, temp), parent(temp, y)))

# Check for sibling relationship between 'a' and 'b'  
def sibling(x, y):
    temp = var()
    return conde((parent(temp, x), parent(temp, y)))

# Check if x is y's uncle
def uncle(x, y):
    temp = var()
    return conde((father(temp, x), grandparent(temp, y)))

if __name__=='__main__':
    father = Relation()
    mother = Relation()
    
    with open('relationships.json') as f:
        d = json.loads(f.read())

    for item in d['father']:
        facts(father, (list(item.keys())[0], list(item.values())[0]))

    for item in d['mother']:
        facts(mother, (list(item.keys())[0], list(item.values())[0]))

    x = var()

    # John's children
    name = 'John'
    output = run(0, x, father(name, x))
    print("\nList of " + name + "'s children:")
    for item in output:
        print(item)

    # William's mother
    name = 'William'
    output = run(0, x, mother(x, name))[0]
    print("\n" + name + "'s mother:\n" + output)

    # Adam's parents 
    name = 'Adam'
    output = run(0, x, parent(x, name))
    print("\nList of " + name + "'s parents:")
    for item in output:
        print(item)

    # Wayne's grandparents 
    name = 'Wayne'
    output = run(0, x, grandparent(x, name))
    print("\nList of " + name + "'s grandparents:")
    for item in output:
        print(item)

    # Megan's grandchildren 
    name = 'Megan'
    output = run(0, x, grandparent(name, x))
    print("\nList of " + name + "'s grandchildren:")
    for item in output:
        print(item)

    # David's siblings 
    name = 'David'
    output = run(0, x, sibling(x, name))
    siblings = [x for x in output if x != name]
    print("\nList of " + name + "'s siblings:")
    for item in siblings:
        print(item)

    # Tiffany's uncles
    name = 'Tiffany'
    name_father = run(0, x, father(x, name))[0]
    output = run(0, x, uncle(x, name))
    output = [x for x in output if x != name_father]
    print("\nList of " + name + "'s uncles:")
    for item in output:
        print(item)

    # All spouses
    a, b, c = var(), var(), var()
    output = run(0, (a, b), (father, a, c), (mother, b, c))
    print("\nList of all spouses:")
    for item in output:
        print('Husband:', item[0], '<==> Wife:', item[1])
     