import json

check_ingredients = False
if check_ingredients:
    out_name = 'salad2.txt'
else:
    out_name = 'salad1.txt'

f = open('layer1.json', 'r')
g = open(out_name, 'w')

parsed_json = json.load(f)
for i, recipe in enumerate(parsed_json):
    #if i % 100 == 0:
    #    print(i)
    found_salad = False
    if 'salad' in recipe['title'].lower():
        print(recipe['title'])
        found_salad = True
    elif check_ingredients:
        for ingr in recipe['ingredients']:
            if 'salad' in ingr['text'].lower():
                print(recipe['title'])
                print(ingr['text'])
                print()
                found_salad = True
                break
    if found_salad:
        g.write(str(recipe['id']) + '\n')

f.close()
g.close()
