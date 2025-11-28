grammar PS;

options {
    language=Python2;
}

WS: [ \t\r\n]+ -> skip;
DOLLAR_SIGN: '\\$' -> skip;

ADD: '+';
SUB: '-';
MUL: '*';
DIV: '/' | '\\over';

L_PAREN: '(';
R_PAREN: ')';
L_GROUP: '\\lgroup';
R_GROUP: '\\rgroup';
L_BRACE: '{';
R_BRACE: '}';
L_BRACE_VISUAL: '\\{';
R_BRACE_VISUAL: '\\}';
L_BRACE_CMD: '\\lbrace';
R_BRACE_CMD: '\\rbrace';
L_BRACKET: '[';
R_BRACKET: ']';
L_BRACK: '\\lbrack';
R_BRACK: '\\rbrack';

BAR: '|';
L_VERT: '\\lvert';
R_VERT: '\\rvert';
VERT: '\\vert';

NORM: '\\|';

L_FLOOR: '\\lfloor';
R_FLOOR: '\\rfloor';
LL_CORNER: '\\llcorner';
LR_CORNER: '\\lrcorner';

L_CEIL: '\\lceil';
R_CEIL: '\\rceil';
UL_CORNER: '\\ulcorner';
UR_CORNER: '\\urcorner';

L_LEFT: '\\left';
R_RIGHT: '\\right';
ML_LEFT: '\\mleft';
MR_RIGHT: '\\mright';

//functions
FUNC_LIM:  '\\lim';
LIM_APPROACH_SYM: '\\to' | '\\rightarrow' | '\\Rightarrow' | '\\longrightarrow' | '\\Longrightarrow';
FUNC_INT:  '\\int';
FUNC_SUM:  '\\sum';
FUNC_PROD: '\\prod';

FUNC_LOG:  '\\log';
FUNC_LN:   '\\ln';
FUNC_EXP: '\\exp';
FUNC_SIN:  '\\sin';
FUNC_COS:  '\\cos';
FUNC_TAN:  '\\tan';
FUNC_CSC:  '\\csc';
FUNC_SEC:  '\\sec';
FUNC_COT:  '\\cot';

FUNC_ARCSIN: '\\arcsin';
FUNC_ARCCOS: '\\arccos';
FUNC_ARCTAN: '\\arctan';
FUNC_ARCCSC: '\\arccsc';
FUNC_ARCSEC: '\\arcsec';
FUNC_ARCCOT: '\\arccot';

FUNC_SINH: '\\sinh';
FUNC_COSH: '\\cosh';
FUNC_TANH: '\\tanh';
FUNC_ARSINH: '\\arsinh';
FUNC_ARCOSH: '\\arcosh';
FUNC_ARTANH: '\\artanh';
FUNC_ARCSINH: '\\arcsinh';
FUNC_ARCCOSH: '\\arccosh';
FUNC_ARCTANH: '\\arctanh';

FUNC_ARSINH_NAME: 'arsinh';
FUNC_ARCSINH_NAME: 'arcsinh';
FUNC_ARCOSH_NAME: 'arcosh';
FUNC_ARCCOSH_NAME: 'arccosh';
FUNC_ARTANH_NAME: 'artanh';
FUNC_ARCTANH_NAME: 'arctanh';
FUNC_GCD_NAME: 'gcd';
FUNC_LCM_NAME: 'lcm';
FUNC_FLOOR_NAME: 'floor';
FUNC_CEIL_NAME: 'ceil';

FUNC_SQRT: '\\sqrt';
FUNC_GCD: '\\gcd';
FUNC_LCM: '\\lcm';
FUNC_FLOOR: '\\floor';
FUNC_CEIL: '\\ceil';
FUNC_MAX: '\\max';
FUNC_MIN: '\\min';

FUNC_DET: '\\det';

FUNC_EYE_NAME: 'eye';
FUNC_ZEROS_NAME: 'zeros';
FUNC_ONES_NAME: 'ones';
FUNC_COLS_NAME: 'cols';
FUNC_ROWS_NAME: 'rows';
FUNC_DIAG_NAME: 'diag';
FUNC_NORM_NAME: 'norm';
FUNC_RANK_NAME: 'rank';
FUNC_TRACE_NAME: 'trace' | 'tr';
FUNC_RREF_NAME: 'rref';
FUNC_HSTACK_NAME: 'hstack';
FUNC_VSTACK_NAME: 'vstack';
FUNC_ORTHOGONALIZE_NAME: 'orth' | 'ortho' | 'orthogonal' | 'orthogonalize';
FUNC_NULLSPACE_NAME: 'nullspace';
FUNC_DIAGONALIZE_NAME: 'eig' | 'eigen' | 'diagonalize';
FUNC_EIGENVALS_NAME: 'eigenvals' | 'eigenvalues';
FUNC_EIGENVECTORS_NAME: 'eigenvects' | 'eigenvectors';
FUNC_SVD_NAME: 'svd' | 'SVD';

//commands
CMD_TIMES: '\\times';
CMD_CDOT:  '\\cdot';
CMD_DIV:   '\\div';
CMD_FRAC:  '\\frac';
CMD_BINOM: '\\binom' | '\\tbinom' | '\\dbinom';
CMD_CHOOSE: '\\choose';
CMD_MOD: '\\mod';

CMD_MATHIT: '\\mathit';

CMD_OPERATORNAME: '\\operatorname';

//matrix test
MATRIX_TYPE_MATRIX: 'matrix';
MATRIX_TYPE_PMATRIX: 'pmatrix';
MATRIX_TYPE_BMATRIX: 'bmatrix';
MATRIX_TYPE_DET: 'vmatrix';
MATRIX_TYPES: MATRIX_TYPE_MATRIX | MATRIX_TYPE_PMATRIX | MATRIX_TYPE_BMATRIX;
CMD_MATRIX_START: '\\begin' L_BRACE MATRIX_TYPES R_BRACE;
CMD_MATRIX_END: '\\end' L_BRACE MATRIX_TYPES R_BRACE;
CMD_DET_START: '\\begin' L_BRACE MATRIX_TYPE_DET R_BRACE;
CMD_DET_END: '\\end' L_BRACE MATRIX_TYPE_DET R_BRACE;
MATRIX_DEL_COL: '&';
MATRIX_DEL_ROW: '\\\\';

UNDERSCORE: '_';
CARET: '^';
COLON: ':';
SEMICOLON: ';';
COMMA: ',';
PERIOD: '.';

fragment WS_CHAR: [ \t\r\n];
DIFFERENTIAL: 'd' WS_CHAR*? ([a-zA-Z] | '\\' [a-zA-Z]+);

EXP_E: 'e' | '\\exponentialE';
E_NOTATION_E: 'E';
LETTER_NO_E: [a-df-zA-DF-Z]; // exclude e for exponential function and e notation
fragment LETTER: [a-zA-Z];
fragment DIGIT: [0-9];

MATRIX_XRIGHTARROW: '\\xrightarrow' | '\\xRightarrow';
TRANSFORM_EXCHANGE: '<->' | '<=>' | '\\leftrightarrow' | '\\Leftrightarrow';

NUMBER:
    DIGIT+ (COMMA DIGIT DIGIT DIGIT)*
    | DIGIT* (COMMA DIGIT DIGIT DIGIT)* PERIOD DIGIT+;

E_NOTATION: NUMBER E_NOTATION_E (SUB | ADD)? DIGIT+;

IN: '\\in';
ASSIGNMENT: '=';
EQUAL: '==' | '\\equiv';
LT: '<';
LTE: '\\leq' | '\\le' | '\\leqslant';
GT: '>';
GTE: '\\geq' | '\\ge' | '\\geqslant';
UNEQUAL: '!=' | '!==' | '\\ne' | '\\neq' | '\\not\\equiv';

BANG: '!';

fragment PERCENT_SIGN: '\\%';
PERCENT_NUMBER: NUMBER PERCENT_SIGN;

//Excludes some letters for use as e.g. constants in SYMBOL
fragment GREEK_LETTER:
    '\\char"000391' | //Alpha
    '\\alpha' |
    '\\char"000392' | //Beta
    '\\beta' |
    '\\Gamma' |
    '\\gamma' |
    '\\Delta' |
    '\\delta' |
    '\\char"000190' | //Epsilon
    '\\epsilon' |
    '\\varepsilon' |
    '\\char"000396' | //Zeta
    '\\zeta' |
    '\\char"000397' | //Eta
    '\\eta' |
    '\\Theta' |
    '\\theta' |
    '\\vartheta' |
    '\\char"000399' | //Iota
    '\\iota' |
    '\\char"00039A' | //Kappa
    '\\kappa' |
    '\\Lambda' |
    '\\lambda' |
    '\\char"00039C' | //Mu
    '\\mu' |
    '\\char"00039D' | //Nu
    '\\nu' |
    '\\Xi' |
    '\\xi' |
    '\\char"00039F' | //Omicron
    '\\omicron' |
    '\\Pi' |
    '\\varpi' |
    '\\char"0003A1' | //Rho
    '\\rho' |
    '\\varrho' |
    '\\Sigma' |
    '\\sigma' |
    '\\varsigma' |
    '\\char"0003A4' | //Tau
    '\\tau' |
    '\\Upsilon' |
    '\\upsilon' |
    '\\Phi' |
    '\\phi' |
    '\\varphi' |
    '\\char"0003A7' | //Chi
    '\\chi' |
    '\\Psi' |
    '\\psi' |
    '\\Omega' |
    '\\omega';

GREEK_CMD: GREEK_LETTER [ ]?;

fragment OTHER_SYMBOL:
    '\\Bbbk'  |
    '\\wp'  |
    '\\nabla'  |
    '\\bigstar'  |
    '\\angle'  |
    '\\nexists'  |
    '\\diagdown'  |
    '\\measuredangle'  |
    '\\eth'  |
    '\\emptyset'  |
    '\\diagup'  |
    '\\sphericalangle'  |
    '\\clubsuit'  |
    '\\varnothing'  |
    '\\Diamond'  |
    '\\complement'  |
    '\\diamondsuit'  |
    '\\imath'  |
    '\\Finv'  |
    '\\triangledown'  |
    '\\heartsuit'  |
    '\\jmath'  |
    '\\Game'  |
    '\\triangle'  |
    '\\spadesuit'  |
    '\\ell'  |
    '\\hbar'  |
    '\\vartriangle'  |
    '\\hslash'  |
    '\\blacklozenge'  |
    '\\lozenge'  |
    '\\blacksquare'  |
    '\\mho'  |
    '\\blacktriangle'  |
    '\\sharp'  |
    '\\prime'  |
    '\\Im'  |
    '\\flat'  |
    '\\square'  |
    '\\backprime'  |
    '\\Re'  |
    '\\natural'  |
    '\\surd'  |
    '\\circledS';
OTHER_SYMBOL_CMD: OTHER_SYMBOL [ ]?;

fragment PI: '\\pi';
fragment INFTY_CMD: '\\infty';
fragment PARTIAL_CMD: '\\partial';
fragment INFTY: INFTY_CMD | DOLLAR_SIGN INFTY_CMD | INFTY_CMD PERCENT_SIGN;
fragment EMPTYSET: '\\emptyset';
SYMBOL: PI | PARTIAL_CMD | INFTY | EMPTYSET;

fragment VARIABLE_CMD: '\\variable';
fragment VARIABLE_SYMBOL: (GREEK_CMD | OTHER_SYMBOL_CMD | LETTER | DIGIT)+ (UNDERSCORE ((L_BRACE (GREEK_CMD | OTHER_SYMBOL_CMD | LETTER | DIGIT | COMMA)+ R_BRACE) | (GREEK_CMD | OTHER_SYMBOL_CMD | LETTER | DIGIT)))?;
VARIABLE: VARIABLE_CMD L_BRACE VARIABLE_SYMBOL R_BRACE PERCENT_SIGN?;

//collection of accents
accent_symbol:
    '\\acute'  |
    '\\bar'  |
    '\\overline'  |
    '\\breve'  |
    '\\check'  |
    '\\widecheck'  |
    '\\dot'  |
    '\\ddot'  |
    '\\grave'  |
    '\\hat'  |
    '\\tilde'  |
    '\\widetilde'  |
    '\\vec'  |
    '\\overrightarrow'  |
    '\\bm'  |
    '\\boldsymbol'  |
    '\\text'  |
    '\\textit'  |
    '\\mathbb'  |
    '\\mathbin'  |
    '\\mathbf'  |
    '\\mathcal'  |
    '\\mathclap'  |
    '\\mathclose'  |
    '\\mathellipsis'  |
    '\\mathfrak'  |
    '\\mathinner'  |
    '\\mathit'  |
    '\\mathnormal'  |
    '\\mathop'  |
    '\\mathopen'  |
    '\\mathord'  |
    '\\mathpunct'  |
    '\\mathrel'  |
    '\\mathring'  |
    '\\mathrlap'  |
    '\\mathrm'  |
    '\\mathscr'  |
    '\\mathsf'  |
    '\\mathsterling'  |
    '\\mathtt';

math: relation | relation_list;

transpose: '^T' | '^{T}' |  '^{\\top}' | '\'';

transform_atom: LETTER_NO_E UNDERSCORE (NUMBER | L_BRACE NUMBER R_BRACE);
transform_scale: (expr | group | ADD | SUB) transform_atom;
transform_swap: transform_atom TRANSFORM_EXCHANGE transform_atom;
transform_assignment: transform_atom transform_scale;
elementary_transform: transform_assignment | transform_scale | transform_swap;
elementary_transforms: elementary_transform (COMMA elementary_transform)*;

matrix:
    CMD_MATRIX_START
    matrix_row (MATRIX_DEL_ROW matrix_row)* MATRIX_DEL_ROW?
    CMD_MATRIX_END
    (MATRIX_XRIGHTARROW (L_BRACKET elementary_transforms R_BRACKET)? L_BRACE elementary_transforms R_BRACE)?;

det:
    CMD_DET_START
    matrix_row (MATRIX_DEL_ROW matrix_row)* MATRIX_DEL_ROW?
    CMD_DET_END;

matrix_row:
    expr (MATRIX_DEL_COL expr)*;

relation:
    relation (IN | ASSIGNMENT | EQUAL | LT | LTE | GT | GTE | UNEQUAL) relation
    | expr;

relation_list:
    relation_list_content
    | L_BRACKET relation_list_content R_BRACKET
    | L_BRACE relation_list_content R_BRACE
    | L_BRACE_VISUAL relation_list_content R_BRACE_VISUAL
    | L_LEFT L_BRACKET relation_list_content R_RIGHT R_BRACKET
    | L_LEFT L_BRACE_VISUAL relation_list_content R_RIGHT R_BRACE_VISUAL
    | ML_LEFT L_BRACKET relation_list_content MR_RIGHT R_BRACKET
    | ML_LEFT L_BRACE_VISUAL relation_list_content MR_RIGHT R_BRACE_VISUAL;

relation_list_content:
    relation COMMA relation (COMMA relation)*
    | relation SEMICOLON relation (SEMICOLON relation)*;

equality:
    expr (EQUAL | ASSIGNMENT) expr;

expr: additive;

additive:
    additive (ADD | SUB) additive
    | mp;

// mult part
mp:
    mp (MUL | CMD_TIMES | CMD_CDOT | DIV | CMD_DIV | COLON | CMD_MOD) mp
    | unary;

mp_nofunc:
    mp_nofunc (MUL | CMD_TIMES | CMD_CDOT | DIV | CMD_DIV | COLON | CMD_MOD) mp_nofunc
    | unary_nofunc;

unary:
    (ADD | SUB) unary
    | postfix+;

unary_nofunc:
    (ADD | SUB) unary_nofunc
    | postfix postfix_nofunc*;

postfix: exp postfix_op*;
postfix_nofunc: exp_nofunc postfix_op*;
postfix_op: BANG | eval_at | transpose;

eval_at:
    BAR (eval_at_sup | eval_at_sub | eval_at_sup eval_at_sub);

eval_at_sub:
    UNDERSCORE L_BRACE
    (expr | equality)
    R_BRACE;

eval_at_sup:
    CARET L_BRACE
    (expr | equality)
    R_BRACE;

exp:
    exp CARET (atom | L_BRACE expr R_BRACE) subexpr?
    | comp;

exp_nofunc:
    exp_nofunc CARET (atom | L_BRACE expr R_BRACE) subexpr?
    | comp_nofunc;

comp:
    group
    | norm_group
    | abs_group
    | floor_group
    | ceil_group
    | func
    | atom
    | frac
    | binom
    | matrix
    | det;

comp_nofunc:
    group
    | norm_group
    | abs_group
    | floor_group
    | ceil_group
    | atom
    | frac
    | binom
    | matrix
    | det;

group:
    L_PAREN expr R_PAREN
    | L_GROUP expr R_GROUP
    | L_BRACE expr R_BRACE
    | L_BRACE_VISUAL expr R_BRACE_VISUAL
    | L_BRACE_CMD expr R_BRACE_CMD
    | L_BRACKET expr R_BRACKET
    | L_BRACK expr R_BRACK
    | L_LEFT L_PAREN expr R_RIGHT R_PAREN
    | L_LEFT L_GROUP expr R_RIGHT R_GROUP
    | L_LEFT L_BRACE expr R_RIGHT R_BRACE
    | L_LEFT L_BRACE_VISUAL expr R_RIGHT R_BRACE_VISUAL
    | L_LEFT L_BRACE_CMD expr R_RIGHT R_BRACE_CMD
    | L_LEFT L_BRACKET expr R_RIGHT R_BRACKET
    | L_LEFT L_BRACK expr R_RIGHT R_BRACK
    | ML_LEFT L_PAREN expr MR_RIGHT R_PAREN
    | ML_LEFT L_GROUP expr MR_RIGHT R_GROUP
    | ML_LEFT L_BRACE expr MR_RIGHT R_BRACE
    | ML_LEFT L_BRACE_VISUAL expr MR_RIGHT R_BRACE_VISUAL
    | ML_LEFT L_BRACE_CMD expr MR_RIGHT R_BRACE_CMD
    | ML_LEFT L_BRACKET expr MR_RIGHT R_BRACKET
    | ML_LEFT L_BRACK expr MR_RIGHT R_BRACK;


norm_group:
    NORM expr NORM
    | L_LEFT NORM expr R_RIGHT NORM
    | ML_LEFT NORM expr MR_RIGHT NORM;


abs_group:
    BAR expr BAR
    | L_VERT expr R_VERT
    | VERT expr VERT
    | L_LEFT BAR expr R_RIGHT BAR
    | L_LEFT L_VERT expr R_RIGHT R_VERT
    | L_LEFT VERT expr R_RIGHT VERT
    | ML_LEFT BAR expr MR_RIGHT BAR
    | ML_LEFT L_VERT expr MR_RIGHT R_VERT
    | ML_LEFT VERT expr MR_RIGHT VERT;


floor_group:
    L_FLOOR expr R_FLOOR
    | LL_CORNER expr LR_CORNER
    | L_LEFT L_FLOOR expr R_RIGHT R_FLOOR
    | L_LEFT LL_CORNER expr R_RIGHT LR_CORNER
    | ML_LEFT L_FLOOR expr MR_RIGHT R_FLOOR
    | ML_LEFT LL_CORNER expr MR_RIGHT LR_CORNER;


ceil_group:
    L_CEIL expr R_CEIL
    | UL_CORNER expr UR_CORNER
    | L_LEFT L_CEIL expr R_RIGHT R_CEIL
    | L_LEFT UL_CORNER expr R_RIGHT UR_CORNER
    | ML_LEFT L_CEIL expr MR_RIGHT R_CEIL
    | ML_LEFT UL_CORNER expr MR_RIGHT UR_CORNER;


//indicate an accent
accent:
    accent_symbol
    L_BRACE base=expr R_BRACE;

atom_expr_no_supexpr: (LETTER_NO_E | GREEK_CMD | OTHER_SYMBOL_CMD | accent) subexpr?;
atom_expr: (LETTER_NO_E | GREEK_CMD | OTHER_SYMBOL_CMD | accent) (supexpr subexpr | subexpr supexpr | subexpr | supexpr)?;
atom: atom_expr | SYMBOL | NUMBER | PERCENT_NUMBER | E_NOTATION | DIFFERENTIAL | mathit | VARIABLE;

mathit: CMD_MATHIT L_BRACE mathit_text R_BRACE;
mathit_text: (LETTER_NO_E | E_NOTATION_E | EXP_E)+;

frac:
    CMD_FRAC L_BRACE
    upper=expr
    R_BRACE L_BRACE
    lower=expr
    R_BRACE;

//a binomial expression
binom:
    L_BRACE upper=expr CMD_CHOOSE lower=expr R_BRACE
    | CMD_BINOM L_BRACE upper=expr R_BRACE L_BRACE lower=expr R_BRACE;

func_normal_functions_single_arg:
    FUNC_LOG | FUNC_LN | FUNC_EXP
    | FUNC_SIN | FUNC_COS | FUNC_TAN
    | FUNC_CSC | FUNC_SEC | FUNC_COT
    | FUNC_ARCSIN | FUNC_ARCCOS | FUNC_ARCTAN
    | FUNC_ARCCSC | FUNC_ARCSEC | FUNC_ARCCOT
    | FUNC_SINH | FUNC_COSH | FUNC_TANH
    | FUNC_ARSINH | FUNC_ARCOSH | FUNC_ARTANH
    | FUNC_ARCSINH | FUNC_ARCCOSH | FUNC_ARCTANH
    | FUNC_FLOOR | FUNC_CEIL | FUNC_DET;

func_normal_functions_multi_arg:
    FUNC_GCD | FUNC_LCM | FUNC_MAX | FUNC_MIN;

func_operator_names_single_arg:
    FUNC_ARSINH_NAME | FUNC_ARCOSH_NAME | FUNC_ARTANH_NAME
    | FUNC_ARCSINH_NAME | FUNC_ARCCOSH_NAME | FUNC_ARCTANH_NAME
    | FUNC_FLOOR_NAME | FUNC_CEIL_NAME | FUNC_EYE_NAME | FUNC_RANK_NAME | FUNC_TRACE_NAME
    | FUNC_RREF_NAME | FUNC_NULLSPACE_NAME | FUNC_DIAGONALIZE_NAME | FUNC_NORM_NAME
    | FUNC_EIGENVALS_NAME | FUNC_EIGENVECTORS_NAME | FUNC_SVD_NAME | FUNC_COLS_NAME | FUNC_ROWS_NAME;

func_operator_names_multi_arg:
    FUNC_GCD_NAME | FUNC_LCM_NAME | FUNC_ZEROS_NAME | FUNC_ORTHOGONALIZE_NAME
    | FUNC_ONES_NAME | FUNC_DIAG_NAME | FUNC_HSTACK_NAME | FUNC_VSTACK_NAME;

func_normal_single_arg:
    (func_normal_functions_single_arg)
    |
    (CMD_OPERATORNAME L_BRACE func_operator_name=func_operator_names_single_arg R_BRACE);

func_normal_multi_arg:
    (func_normal_functions_multi_arg)
    |
    (CMD_OPERATORNAME L_BRACE func_operator_name=func_operator_names_multi_arg R_BRACE);

func:
    func_normal_single_arg
    (subexpr? supexpr? | supexpr? subexpr?)
    (L_LEFT? L_PAREN func_single_arg R_RIGHT? R_PAREN | ML_LEFT? L_PAREN func_single_arg MR_RIGHT? R_PAREN | func_single_arg_noparens)
    
    | func_normal_multi_arg
    (subexpr? supexpr? | supexpr? subexpr?)
    (L_LEFT? L_PAREN func_multi_arg R_RIGHT? R_PAREN | ML_LEFT? L_PAREN func_multi_arg MR_RIGHT? R_PAREN | func_multi_arg_noparens)

    | atom_expr_no_supexpr supexpr?
    L_LEFT? (L_PAREN | L_BRACKET) func_common_args (R_PAREN | R_BRACKET) R_RIGHT?
    | atom_expr_no_supexpr supexpr?
    L_BRACE L_LEFT? (L_PAREN | L_BRACKET) func_common_args (R_PAREN | R_BRACKET) R_RIGHT? R_BRACE

    | FUNC_INT
    (subexpr supexpr | supexpr subexpr | (UNDERSCORE L_BRACE R_BRACE) (CARET L_BRACE R_BRACE) | (CARET L_BRACE R_BRACE) (UNDERSCORE L_BRACE R_BRACE) )?
    (additive? DIFFERENTIAL | frac | additive)

    | FUNC_SQRT
    (L_BRACKET root=expr R_BRACKET)?
    L_BRACE base=expr R_BRACE

    | (FUNC_SUM | FUNC_PROD)
    (subeq supexpr | supexpr subeq)
    mp
    | FUNC_LIM limit_sub mp
    | EXP_E supexpr?; //Exponential function e^x

args: (expr ',' args) | expr;

func_common_args: atom | (expr ',') | (expr ',' args);

limit_sub:
    UNDERSCORE L_BRACE
    (LETTER_NO_E | GREEK_CMD | OTHER_SYMBOL_CMD)
    LIM_APPROACH_SYM
    expr (CARET L_BRACE (ADD | SUB) R_BRACE)?
    R_BRACE;

func_single_arg: expr;
func_single_arg_noparens: mp_nofunc;

func_multi_arg: expr | (expr ',' func_multi_arg);
func_multi_arg_noparens: mp_nofunc;

subexpr: UNDERSCORE (atom | L_BRACE (expr | args) R_BRACE);
supexpr: CARET (atom | L_BRACE expr R_BRACE);

subeq: UNDERSCORE L_BRACE equality R_BRACE;
supeq: UNDERSCORE L_BRACE equality R_BRACE;
