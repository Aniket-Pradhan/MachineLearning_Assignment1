# a = "age: continuous.\n\
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.\n\
# fnlwgt: continuous.\n\
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.\n\
# education-num: continuous.\n\
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.\n\
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.\n\
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.\n\
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.\n\
# sex: Female, Male.\n\
# capital-gain: continuous.\n\
# capital-loss: continuous.\n\
# hours-per-week: continuous.\n\
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands."

# # print(a.split("\n"))
# a = a.split("\n")
# label = []
# for i in a:
#     # print(i.split(":")[0])
#     label.append(i.split(":")[0])

# print(len(label))

# for i in label:
#     print("\"" + i + "\",", end='')

# print()

a = input()
a = a.split(", ")
print("{", end="")
for i in range(len(a)):
    print("\"" + str(a[i] + "\": " + str(i) + ", "), end="")