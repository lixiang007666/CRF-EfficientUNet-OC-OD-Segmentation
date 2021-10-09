import re


with open('text.txt','r',encoding='utf-8') as f:
    str1 = f.read()
print(str1)
train= re.findall(r"train loss : (.+?)\n2021", str1)
print(train)
train_post=[]
for i in train:
    train_post.append(1-float(i.strip()[1:]))
print(train_post)
validation = re.findall(r"validation loss: (.+?)\n2021", str1)
print(validation)
validation_post=[]
for i in validation:
    validation_post.append(1-float(i.strip()[1:]))
print(validation_post)

dice = re.findall(r"Average global foreground Dice: (.+?)\n2021", str1)
print(dice)
dice_post_temp=[]
for i in dice:
    dice_post_temp.append(i.rstrip('] ').lstrip('['))
print(dice_post_temp)
dice1=[]
dice2=[]
for i in dice_post_temp:
    dice1.append(float(i.split(', ')[0]))
    dice2.append(float(i.split(', ')[1]))
print(dice1)
print(dice2)
