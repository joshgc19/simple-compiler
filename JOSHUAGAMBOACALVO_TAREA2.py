import math
import re
import datetime

# CURRENT DATE
date = datetime.datetime.now()

# CONSTANT VARIABLES
SYMBOLS = [',', '(', ')', '+', '-', '^', '/', '*', '=', '<', '>', ':', ';', '!']
RESERVED_WORDS = ['fun', 'if', 'then', 'else', 'load']
CONSTANTS = ['pi', 'e']
MATH_FUNCTIONS = ['ln', 'tan', 'sin', 'cos']

# DYNAMIC VARIABLES
variables = {}
userInput = ""
command: list = []
parsed_instructions: list = []

# FUNCTION THAT LOAD AN ARCHIVE ADDING THE .SIM EXTENSION TO IT
def load_archive(archive_name: str) -> str:
    try:
        f = open(f'{archive_name}.sim', 'r')
    except IOError:
        raise Exception(f"IO Exception: File not found.\n\tFile named {archive_name}.sim doesn't exist, please ensure the file has the correct extension and it's in the same directory as the current program.")
    else:
        return ''.join([line for line in f])



# Node class that models a compiler node
class Node:

    def __init__(self, token: str, kind: str, line: int, col: int):
        self.token = token
        self.kind = kind
        self.line = line
        self.col = col

    def __str__(self):
        return f' <|token= {self.token},kind= {self.kind}, line= {self.line}, col= {self.col}|> '

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return type(other) == type(self) and other.token == self.token


# Function that returns the last node in line
def get_token() -> Node | None:
    global command
    if not command:
        return None
    token = command[0]
    command = command[1:]
    return token

# FUNCTION THAT RETURNS A TOKEN TO THE STACK
def ret_token(token):
    global command
    if token:
        command = [token] + command

# FUNCTION THAT RETURNS THE LAST TOKEN IN THE STACK WITHOUT POPPING IT
def peek_token() -> Node | None:
    global command
    if not command:
        return None
    return command[0]


# definition simple_rest
def simple():
    grammar_definition = definition()
    grammar_rest = peek_token()
    if not grammar_rest:
        return [grammar_definition]
    else:
        return [grammar_definition] + simple_rest()


# ; simple | e
def simple_rest():
    grammar_dot_comma = get_token()
    if grammar_dot_comma.token == ';':
        return simple()
    else:
        raise Exception(
            f"Syntax error: Invalid syntax at {grammar_dot_comma.line}:{grammar_dot_comma.col}.\n\tCompiler was expecting \";\" but received \"{grammar_dot_comma.token}\".")


# identifier = exp | identifier = fun | exp
def definition():
    grammar_identification = get_token()
    grammar_eq = get_token()
    if grammar_identification:
        if grammar_identification.kind == 'IDENTIFIER':
            check_variable(grammar_identification)
            if grammar_eq:
                if grammar_eq.token == '=':
                    grammar_test = peek_token()
                    if grammar_test and grammar_test.token == 'fun':
                        return [grammar_eq, grammar_identification, fun()]
                    else:
                        return [grammar_eq, grammar_identification, exp()]
                else:
                    ret_token(grammar_eq)
                    ret_token(grammar_identification)
                    return exp()
            else:
                ret_token(grammar_identification)
                return exp()
        elif grammar_identification.kind in ['CONSTANT', 'RESERVED_WORD']:
            if grammar_eq is not None and grammar_eq.token == '=':
                raise Exception(
                    f"Syntax error: You cannot redefine constants or reserved words at {grammar_eq.line}:{grammar_eq.col}.\n\tRefrain of using the following words to name variables {', '.join(RESERVED_WORDS)} or {', '.join(CONSTANTS)}.")
            else:
                ret_token(grammar_eq)
                ret_token(grammar_identification)
                return exp()
        else:
            ret_token(grammar_eq)
            ret_token(grammar_identification)
            return exp()
    else:
        raise Exception("Opss... You shouldnt be here")


# fun(ident_list) :  exp
def fun():
    grammar_fun = get_token()
    if grammar_fun.token == 'fun':
        grammar_open_parenthesis = get_token()
        if grammar_open_parenthesis and grammar_open_parenthesis.token == '(':
            grammar_ident_list = ident_list(grammar_fun)
            grammar_close_parenthesis = get_token()
            if grammar_close_parenthesis and grammar_close_parenthesis.token == ')':
                grammar_two_dots = get_token()
                if grammar_two_dots:
                    if grammar_two_dots.token == ':':
                        return [grammar_fun, grammar_ident_list, exp()]
                    else:
                        raise Exception(
                            f"Syntax error: Unexpected token at {grammar_two_dots.line}:{grammar_two_dots.col}.\n\tCompiler was expecting \":\" and got \"{grammar_two_dots.token}\".")
                else:
                    raise Exception(
                        f"Syntax error: Invalid syntax at {grammar_fun.line}:{grammar_fun.col}.\n\tFunction with no body.")
            else:
                if grammar_close_parenthesis:
                    raise Exception(
                        f"Syntax error: Unexpected token at {grammar_close_parenthesis.line}:{grammar_close_parenthesis.col}.\n\tCompiler was expecting \")\" and got \"{grammar_close_parenthesis.token}\".")
                else:
                    raise Exception(
                        f"Syntax error: Invalid syntax at {grammar_open_parenthesis.line}:{grammar_open_parenthesis.col}.\n\tMismatch parenthesis.")
        else:
            raise Exception(
                f"Syntax error: Invalid syntax at {grammar_fun.line}:{grammar_fun.col}.\n\tCompiler was expecting \"(\".")
    else:
        ret_token(grammar_fun)

# identifier ident_rest
def ident_list(grammar_fun):
    grammar_ident = get_token()
    if grammar_ident:
        if grammar_ident.kind in ['MATH_FUNCTION', 'RESERVED_WORD', 'CONSTANT']:
            raise Exception(
                f"Syntax error: Invalid syntax at {grammar_ident.line}:{grammar_ident.col}.\n\tCannot use a reserved word as name of a parameter.")
        elif grammar_ident.kind == 'IDENTIFIER':
            return [grammar_ident] + ident_rest(fun)
        else:
            raise Exception(
                f"Syntax error: Unexpected token at {grammar_ident.line}:{grammar_ident.col}.\n\tThe compiler did not recognized the use of the token {grammar_ident.token}.")
    else:
        raise Exception(
            f"Syntax error: Unexpected end of function at {grammar_fun.line}:{grammar_fun.col}.\n\tThe function definition didn't end as expected.")

# , ident_list | e
def ident_rest(grammar_fun):
    grammar_comma = get_token()
    if grammar_comma:
        if grammar_comma.token == ',':
            return ident_list(grammar_fun)
        else:
            ret_token(grammar_comma)
            return []
    else:
        raise Exception(
            f"Syntax error: Unexpected end of function at {grammar_fun.line}:{grammar_fun.col}.\n\tThe function definition didn't end as expected.")


# exp ::= exp + term | exp - term | term
def exp():
    grammar_exp = peek_token()
    if grammar_exp and grammar_exp.token == 'if':
        return If()
    else:
        return formula()

# if condition then exp else exp
def If():
    grammar_if = get_token()
    if grammar_if.token == 'if':
        grammar_condition = condition(grammar_if)
        grammar_then = get_token()
        if grammar_then:
            if grammar_then.token == 'then':
                grammar_exp = exp()
                grammar_else = get_token()
                if grammar_else:
                    if grammar_else.token == 'else':
                        grammar_exp_2 = exp()
                        return [grammar_if, grammar_condition, grammar_exp, grammar_exp_2]
                    else:
                        raise Exception(
                            f"Syntax error: Unexpected token at {grammar_else.line}:{grammar_else.col}.\n\tCompiler was expecting \"else\" and got \"{grammar_else.token}\".")
                else:
                    raise Exception(
                        f"Syntax error: Unexpected end of if at {grammar_if.line}:{grammar_if.col}.\n\tIncorrect if grammar, refer to \"if <condition> then <expression> else <expression>\".")
            else:
                raise Exception(
                    f"Syntax error: Unexpected token at {grammar_then.line}:{grammar_then.col}.\n\tCompiler was expecting \"then\" and got \"{grammar_then.token}\".")
        else:
            raise Exception(
                f"Syntax error: Unexpected end of if at {grammar_if.line}:{grammar_if.col}.\n\tIncorrect if grammar, refer to \"if <condition> then <expression> else <expression>\".")
    else:
        raise Exception('DEV ERROR')

# exp logic_operator exp
def condition(grammar_if):
    grammar_exp = exp()
    grammar_op = logic_operator(grammar_if)
    grammar_exp_2 = exp()
    return [grammar_op, grammar_exp, grammar_exp_2]

# == | > | != | >= | <=
def logic_operator(grammar_if):
    grammar_op = get_token()
    if grammar_op:
        grammar_op_2 = get_token()
        if grammar_op.token in ['>', '<', '!', '=']:
            if grammar_op_2:
                if grammar_op_2.token == '=':
                    grammar_op.token += grammar_op_2.token
                    return grammar_op
                else:
                    if grammar_op.token in ['<', '>']:
                        ret_token(grammar_op_2)
                        return grammar_op
                    else:
                        raise Exception(
                            f"Syntax error: Unexpected token at {grammar_op_2.line}:{grammar_op_2.col}.\n\tCompiler was expecting \"=\" and got \"{grammar_op_2.token}\".")
            else:
                raise Exception(
                    f"Syntax error: Unexpected end of if at {grammar_if.line}:{grammar_if.col}.\n\tIncorrect if grammar, refer to \"if <condition> then <expression> else <expression>\".")
        else:
            raise Exception(
                f"Syntax error: Unexpected token at {grammar_op.line}:{grammar_op_2.col}.\n\tCompiler was expecting logic operator and got \"{grammar_op.token}\".")
    else:
        raise Exception(
            f"Syntax error: Unexpected end of if at {grammar_if.line}:{grammar_if.col}.\n\tIncorrect if grammar, refer to \"if <condition> then <expression> else <expression>\".")

# formula + term | formula - term | term
def formula():
    grammar_term = term()
    grammar_op = get_token()
    while grammar_op and (grammar_op.token == '+' or grammar_op.token == '-'):
        grammar_term = [grammar_op, grammar_term, term()]
        grammar_op = get_token()
    ret_token(grammar_op)
    return grammar_term


# term ::= term * fact | term / fact | fact
def term():
    grammar_fact = fact()
    grammar_op = get_token()
    while grammar_op and (grammar_op.token == '*' or grammar_op.token == '/'):
        grammar_fact = [grammar_op, grammar_fact, fact()]
        grammar_op = get_token()
    ret_token(grammar_op)
    return grammar_fact


# fact ::= value ^ fact | value
def fact():
    grammar_value = value()
    grammar_op = get_token()
    if grammar_op and grammar_op.token == "^":
        grammar_value = [grammar_op, grammar_value, fact()]
    else:
        ret_token(grammar_op)

    return grammar_value


# value ::= (exp) | variable | fun (exp) | num | const
def value():
    grammar_op = get_token()
    if grammar_op and grammar_op.token == '(':
        grammar_expression = exp()
        grammar_op_closing = get_token()
        if not grammar_op_closing or grammar_op_closing.token != ')':
            raise Exception("Syntax error: Invalid syntax.\n\tMismatch parenthesis.")
        else:
            return grammar_expression
    elif grammar_op and grammar_op.kind == 'MATH_FUNCTION':
        grammar_open_parenthesis = get_token()
        if grammar_open_parenthesis and grammar_open_parenthesis.token == '(':
            grammar_expression = exp()
            grammar_close_parenthesis = get_token()
            if not grammar_close_parenthesis or grammar_close_parenthesis.token != ')':
                raise Exception("Syntax error: Invalid syntax.\n\tMismatch parenthesis.")
            else:
                return [grammar_op, grammar_expression]
        else:
            raise Exception("Syntax error: Invalid syntax.\n\tBad math function syntax.")
    elif grammar_op and grammar_op.kind == 'CONSTANT':
        return grammar_op
    elif grammar_op and grammar_op.kind == 'IDENTIFIER':
        grammar_parenthesis = peek_token()
        if grammar_parenthesis and grammar_parenthesis.token == '(':
            grammar_params = params()
            return [grammar_op, grammar_params]
        return grammar_op
    elif grammar_op and grammar_op.kind == 'NUMBER':
        return float(grammar_op.token)
    elif grammar_op and grammar_op.token == 'load':
        grammar_parenthesis = peek_token()
        if grammar_parenthesis and grammar_parenthesis.token == '(':
            grammar_params = params()
            return [grammar_op, grammar_params]
        return grammar_op
    else:
        if grammar_op:
            raise Exception(
                f"Syntax error: Unexpected token at {grammar_op.line}:{grammar_op.col}.\n\tCompiler was expecting a value or expression and got \"{grammar_op.token}\".")
        else:
            raise Exception(
                f"Syntax error: Unexpected end of line expected a value.")

# (params_list) | e
def params():
    grammar_open_parenthesis = get_token()
    grammar_param_list = param_list(grammar_open_parenthesis)
    grammar_close_parenthesis = get_token()
    if grammar_close_parenthesis:
        if grammar_close_parenthesis.token == ')':
            return grammar_param_list
        else:
            raise Exception(
                f"Syntax error: Unexpected token at {grammar_close_parenthesis.line}:{grammar_close_parenthesis.col}.\n\tMismatch parenthesis, compiler got \"{grammar_close_parenthesis.token}\".")
    else:
        raise Exception(
            f"Syntax error: Unexpected end of function call at {grammar_open_parenthesis.line}:{grammar_open_parenthesis.col}.\n\tCompiler was expecting \")\".")

# exp resto_params
def param_list(grammar_open_parenthesis):
    return [exp()] + resto_params(grammar_open_parenthesis)

# , params_list | e
def resto_params(grammar_open_parenthesis):
    grammar_comma = get_token()
    if grammar_comma:
        if grammar_comma.token == ',':
            return param_list(grammar_open_parenthesis)
        else:
            ret_token(grammar_comma)
            return []
    else:
        raise Exception(
            f"Syntax error: Unexpected end of function call at {grammar_open_parenthesis.line}:{grammar_open_parenthesis.col}.\n\tCompiler was expecting parameter list.")

# FUNCTION THAT CREATES A COMPILER TOKEN CHECKING ITS TYPE
def createToken(token, line, col):
    if token == '':
        raise Exception(f'DEVELOPMENT ERROR: NO TOKEN GOT TO PUSH TOKEN AT LINE {line}:{col}')

    if token in SYMBOLS:
        kind = 'SYMBOL'
    elif token in RESERVED_WORDS:
        kind = 'RESERVED_WORD'
    elif token in CONSTANTS:
        kind = 'CONSTANT'
    elif token in MATH_FUNCTIONS:
        kind = 'MATH_FUNCTION'
    elif re.match('\d+\.$', token):
        raise Exception(
            f"Syntax error: Invalid syntax at {line}:{col}.\n\tA number may not end with a decimal dot.")
    elif re.match('(\d+\.\d+)$|\d+$', token):
        kind = 'NUMBER'
    else:
        if token[-1] == '_':
            raise Exception(
                f"Syntax error: Invalid syntax at {line}:{col}.\n\tA definition may not end with an underscore symbol (_).")
        kind = 'IDENTIFIER'

    return Node(token, kind, line, col - len(token))

# FUNCTION THAT PARSES AN STRING INTO COMPILER NODES
def tokenizer(local_input: str):
    local_tokenized_input = []

    numberFlag = False
    wordFlag = False
    commentFlag = False

    word = ""

    line = 1
    col = 0

    for char in local_input:

        # LOCATION MANAGEMENT
        if char == '\n':
            line += 1
            col = 0
            if word:
                local_tokenized_input.append(createToken(word, line, col))
                word = ''
            wordFlag = commentFlag = numberFlag = False
        elif char == '\t':
            col += 4
            if word:
                local_tokenized_input.append(createToken(word, line, col))
                word = ''
            wordFlag = commentFlag = numberFlag = False
        else:
            col += 1

        # COMMENT MANAGING
        if commentFlag:
            continue

        # COMMENT MANAGING
        if char == '#':
            if word:
                wordFlag = numberFlag = False
                local_tokenized_input.append(createToken(word, line, col))
                word = ''
            commentFlag = True

        # SPACE OR EOF MANAGING
        if char == ' ' or char == '\t':
            if not wordFlag and numberFlag:
                if word.count('.') > 1:
                    raise Exception(
                        f"Syntax error: Invalid syntax at {line}:{col}.\n\tA number may not have more than one decimal dot.")
                elif word.count('.') == 1:
                    if word[-1] == '.':
                        raise Exception(
                            f"Syntax error: Invalid syntax at {line}:{col}.\n\tA number may not end in a floating dot.")

            if word:
                local_tokenized_input.append(createToken(word, line, col))
                word = ''
                wordFlag = numberFlag = False

        # WORD MANAGING
        elif char.isalpha() or char == '_':
            if not wordFlag:
                if numberFlag:
                    raise Exception(
                        f"Syntax error: Unexpected Token at {line}:{col}.\n\tThe compiler did not recognize the usage of the token \"" + char + "\".")
                elif char == '_':
                    raise Exception(
                        f"Syntax error: Invalid syntax at {line}:{col}.\n\tA definition may not start with an underscore symbol (_).")
                wordFlag = True
            word += char

        # NUMBER OR WORD MANAGING
        elif char.isnumeric() or char == '.':
            if wordFlag:
                if char == '.':
                    raise Exception(
                        f"Syntax error: Invalid syntax at {line}:{col}.\n\tA definition may not have a dot(.) in it.")
            elif not numberFlag:
                if char == '.':
                    raise Exception(
                        f"Syntax error: Unexpected Token at {line}:{col}.\n\tThe compiler did not recognize the usage of the token \"" + char + "\".")
                else:
                    numberFlag = True
            word += char

        # SYMBOL DECLARATION
        elif char in SYMBOLS:
            if word:
                wordFlag = numberFlag = False
                local_tokenized_input.append(createToken(word, line, col))
                word = ''
            local_tokenized_input.append(createToken(char, line, col + 1))

        else:
            Exception(
                f"Syntax error: Unexpected Token at {line}:{col}.\n\tThe compiler did not recognize the  token \"" + char + "\".")
    if word:
        local_tokenized_input.append(createToken(word, line, col + 1))
    return local_tokenized_input


# CHECKS IF THE IDENTIFIER IS VALID
def check_variable(variableNode: Node):
    if variableNode.kind in ['RESERVED_WORD', 'MATH_FUNCTION', 'CONSTANT']:
        raise Exception(
            f"Syntax error: Invalid syntax at {variableNode.line}:{variableNode.col}.\n\tYou may not use built-in functions nor constants names as variable names.")
    if variableNode.kind != 'IDENTIFIER':
        raise Exception(
            f"Syntax error: Invalid syntax at {variableNode.line}:{variableNode.col}.\n\tA definition or call must start with a valid identifier.")

# RETRIEVES THE VALUE OF A CONSTANT
def get_constant(node: Node):
    if node.token == 'pi':
        return math.pi
    elif node.token == 'e':
        return math.e
    else:
        raise Exception(
            f"Syntax error: Unrecognized constant at {node.line}:{node.col}.\n\tThe compiler didn't recognized the constant \"{node.token}\".")

# RETRIEVES THE VALUE OF A NODE OR FLOAT GIVEN THE ENVIROMENT
def get_value(node: Node | float, environment: dict | None, idx: int):
    if isinstance(node, float):
        return node
    elif node.kind == 'CONSTANT':
        return get_constant(node)
    elif node.kind == 'IDENTIFIER':
        current_var: Node | None = None
        if environment and node.token in environment.keys():
            current_var = environment[node.token]
        if not current_var and node.token in variables.keys():
            current_var = variables[node.token]
        if current_var is not None:
            if isinstance(current_var, list):
                raise Exception(
                    f"Syntax error: Function call with no parameters {node.line}:{node.col}.\n\tThe variable \"{node.token}\" is a function and was called with no parameters.")
            else:
                return current_var
        else:
            raise Exception(
                f"Syntax error: Variable not found at {node.line}:{node.col}.\n\tThe variable \"{node.token}\" was not found the environment.")
    else:
        raise Exception(
            f"Syntax error: Unexpected token at {node.line}:{node.col}.\n\tThe compiler did not recognized the use of the token \"{node.token}\" and wasn't not found the environment.")

# GETS THE VALUE OF A MATH FUNCTION GIVEN ITS PARAMETERS
def get_math_function_value(expression, environment: dict = None, idx: int = 0):
    if len(expression) != 2:
        raise Exception(
            f"Syntax error: Unknown usage of the math function \"{expression[0].token}\" at {expression[0].line}:{expression[0].col}.\n\tThe correct use of the function is \"{expression[0].token}(<number>)\".")
    if expression[0].token == 'ln':
        return math.log(resolve(expression[1], environment, idx))
    elif expression[0].token == "sin":
        return math.sin(resolve(expression[1], environment, idx))
    elif expression[0].token == "cos":
        return math.cos(resolve(expression[1], environment, idx))
    elif expression[0].token == "tan":
        return math.tan(resolve(expression[1], environment, idx))

# FUNCTION THAT MANAGES THE ASSIGN OF A FUNCTION OR VARIABLE
def manage_assign(expression: list[Node | list[Node]], environment: dict = None, idx: int = 0):
    check_variable(expression[1])
    if isinstance(expression[2], list) and expression[2][0].token == 'fun':
        variables[expression[1].token] = expression[2]
    else:
        variables[expression[1].token] = resolve(expression[2], environment, idx)
        return variables[expression[1].token]

# FUNCTION THAT MANAGES A CONDITION EVALUATION
def manage_condition(expression: list[Node | list[Node]], environment: dict = None, idx: int = 0):
    if expression[0].token == '==':
        return resolve(expression[1], environment, idx) == resolve(expression[2], environment, idx)
    elif expression[0].token == '!=':
        return resolve(expression[1], environment, idx) != resolve(expression[2], environment, idx)
    elif expression[0].token == '>=':
        return resolve(expression[1], environment, idx) >= resolve(expression[2], environment, idx)
    elif expression[0].token == '<=':
        return resolve(expression[1], environment, idx) <= resolve(expression[2], environment, idx)
    elif expression[0].token == '>':
        return resolve(expression[1], environment, idx) > resolve(expression[2], environment, idx)
    elif expression[0].token == '<':
        return resolve(expression[1], environment, idx) < resolve(expression[2], environment, idx)
    else:
        raise Exception("DEV ERROR")

# FUNCTION THAT OPERATES TWO NODES
def get_operation(expression: list[Node], environment: dict = None, idx: int = 0):
    if expression[0].token == '+':
        return resolve(expression[1], environment, idx) + resolve(expression[2], environment, idx)
    elif expression[0].token == '-':
        return resolve(expression[1], environment, idx) - resolve(expression[2], environment, idx)
    elif expression[0].token == '*':
        return resolve(expression[1], environment, idx) * resolve(expression[2], environment, idx)
    elif expression[0].token == '/':
        denominator: int = resolve(expression[2], environment, idx)
        if denominator != 0:
            return resolve(expression[1], environment, idx) / denominator
        else:
            raise Exception(
                f"Math error: Division by zero at {expression[0].line}:{expression[0].col}.\n\tThe denominator of the division is zero so it cannot be divided.")
    elif expression[0].token == '^':
        return resolve(expression[1], environment, idx) ** resolve(expression[2], environment, idx)
    elif expression[0].token == '=':
        return manage_assign(expression, environment, idx)
    else:
        return manage_condition(expression, environment, idx)

# FUNCTION THAT MANAGES RESERVED WORDS SUCK AS IF AND LOAD
def get_reserved_word(expression: list[Node | list[Node]] | Node, environment: dict = None, idx: int = 0):
    global parsed_instructions, command
    if expression[0].token == 'if':
        return resolve(expression[2], environment, idx) if resolve(expression[1], environment, idx) else resolve(
            expression[3], environment, idx)
    elif expression[0].token == 'load':
        if len(expression[1]) == 1:
            file_input = load_archive(expression[1][0].token)
            command = tokenizer(file_input)
            file_parsed_input = simple()
            for i in range(len(file_parsed_input)):
                parsed_instructions.insert(i + idx + 1, file_parsed_input[i])
        else:
            raise Exception(
                f"Function error at function load.\n\tFunction expected 1 positional arguments but {len(expression[1])} were given.")
    else:
        print(expression)
        raise Exception("DEV ERROR")

# FUNCTION THAT
def bound_parameters(parameters, identifiers, environment, function_name):
    if len(parameters) != len(identifiers):
        raise Exception(
            f"Function error at function {function_name}.\n\tFunction expected {len(identifiers)} positional arguments but {len(parameters)} were given.")
    new_environment = {}
    for i, ident in enumerate(identifiers):
        new_environment[ident.token] = resolve(parameters[i], environment)
    return new_environment


def get_function_value(expression: list[Node | list[Node]] | Node, environment: dict = None, idx: int = 0):
    if not expression[0].token in variables.keys():
        raise Exception(
            f"Syntax error: Compiler did not recognized the function at {expression[0].line}:{expression[0].col}.\n\tFunction {expression[0].token} haven't been defined.")
    function_body = variables[expression[0].token]
    if not isinstance(function_body, list):
        raise Exception(
            f"Syntax error: Variable {expression[0].token} is not a function at :{expression[0].col}.\n\tVariable was called as a function.")
    return resolve(function_body[2],
                   bound_parameters(expression[1], function_body[1], environment, expression[0].token), idx)


def resolve(expression: list[Node] | Node, environment: dict = None, idx: int = 0):
    global variables
    if expression is None:
        return
    if not isinstance(expression, list):
        return get_value(expression, environment, idx)
    elif len(expression) == 1:
        return resolve(expression[0], environment, idx)
    elif expression[0].kind == 'SYMBOL':
        return get_operation(expression, environment, idx)
    elif expression[0].kind == 'RESERVED_WORD':
        return get_reserved_word(expression, environment, idx)
    elif expression[0].kind == 'MATH_FUNCTION':
        return get_math_function_value(expression, environment, idx)
    elif expression[0].kind == 'IDENTIFIER':
        return get_function_value(expression, environment, idx)


def init():
    global userInput, command, date, parsed_instructions

    print(
        f'Simple compiler: Release v1.0 Production on {date.strftime("%a")} {date.strftime("%b")} {date.strftime("%d")} {date.strftime("%Y")} {date.strftime("%H")}:{date.strftime("%M")}:{date.strftime("%S")}\n'
        f'Copyright (c) 2001, 2022 Simple. All rights reserved. Author Joshua Gamboa Calvo.\n'
        f'Type \"exit\" to end execution.\n')

    userInput = input('simple> ')

    while userInput != 'exit':
        try:
            command = tokenizer(userInput)
            if command:
                parsed_instructions = simple()

                if len(parsed_instructions) == 1:
                    res = None
                    for tree in parsed_instructions:
                        res = resolve(tree)
                    if res is not None:
                        print(res)
                else:
                    idx = 0
                    for tree in parsed_instructions:
                        resolve(tree, None, idx)
                        idx += 1
        except Exception as e:
            print(str(e))
        command = []
        parsed_instructions = []
        userInput = input('simple> ')


init()
