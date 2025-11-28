from sympy import *
from latex2sympy import process_sympy


#
# Equality Testing
#

answer_sets = [
    {
        'correct_answer': '(x-y)(x+2y)',
        'student_answers': [
            'x^2+xy-2y^2',
            '(x-y)(x+2y)',
            '(x+2y)(x-y)',
            '(2\\times y+x)(-y+x)',
            '(y\\cdot 2+x)(-y+x)'
        ]
    },
    {
        'correct_answer': '2\\pi \\variable{r}^2',
        'student_answers': [
            '2\\pi \\variable{r}^2',
            '\\pi 2\\variable{r}^2',
            '2\\times \\pi \\times \\variable{r}^2',
            '2\\pi \\variable{r} \\times \\variable{r}'
        ]
    },
    {
        'correct_answer': '2x - 3y',
        'student_answers': [
            '-3y + 2x'
        ]
    },
    {
        'correct_answer': 'x\\times x',
        'student_answers': [
            'x\\times x',
            'x\\cdot x',
            'x^2',
            '(\\sqrt{x})^{4}'
        ]
    },
    {
        'correct_answer': '23e^{-1\\times \\sqrt{t^2}}',
        'student_answers': [
            '23e^{-t}'
        ]
    },
    {
        'correct_answer': 'a=x^2+1',
        'student_answers': [
            'x^2+1=a'
        ]
    }
]

for answer_set in answer_sets:
    correct_answer = answer_set['correct_answer']
    correct_answer_parsed = process_sympy(answer_set['correct_answer'])
    for student_answer in answer_set['student_answers']:
        student_answer_parsed = process_sympy(student_answer)
        print('correct_answer (c): ', correct_answer, correct_answer_parsed)
        print('student_answer (a): ', student_answer, student_answer_parsed)
        print('')
        print('Expression Tree (srepr(c) == srepr(a)) =>', srepr(correct_answer_parsed) == srepr(student_answer_parsed))
        print('srepr(c) =>', srepr(correct_answer_parsed))
        print('srepr(a) =>', srepr(student_answer_parsed))
        print('')
        # print('Structural (c == a) =>', correct_answer_parsed == student_answer_parsed)
        print('Symbolic (simplify(c - s) == 0) =>', simplify(correct_answer_parsed - student_answer_parsed) == 0)
        print('simplified =>', simplify(correct_answer_parsed - student_answer_parsed))
        print('')
        print('Numeric Substitution (c.equals(s)) =>', correct_answer_parsed.equals(student_answer_parsed))
        print('-----------------------------------------------------')
