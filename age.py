age = int(input("Enter your age: "))
is_finnish = input("Are you Finnish? (yes/no): ")

if is_finnish == "yes":
    if age >= 18:
        print("you are eligible for the Finnish voting process. ")
    else:
        print("you are not eligible for the finnish voting process because you are a minor.")
else:
        print("you are not a citizen of the country. ")



