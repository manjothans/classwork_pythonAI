def grade_calculator():
    score = float(input("Enter the score: "))
    try:

        if score >= 90:
            grade = 'A'
        elif score >= 80:
            grade = 'B'
        elif score >= 70:
            grade = 'C'
        elif score >= 60:
            grade = 'D'
        else:
            grade = 'E'
        print(f"Grade: {grade}")

 #"if found any invalid input"

    except Exception as e:
        print("Error: Invalid input. Please enter a numeric value.")


grade_calculator()