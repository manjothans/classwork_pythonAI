def medicine():
    age = int(input("Enter the patient's age: "))

    if age >= 18:
        print("The medicine can be given.")
    elif 15 <= age < 18:
        weight = float(input("Enter the patient's weight (kg): "))
        if weight >= 55:
            print("The medicine can be given.")
        else:
            print("The medicine cannot be given .")
    else:
        print("The medicine cannot be given .")


medicine()

