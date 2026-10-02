// Dense layer weights: 4 outputs, 400 inputs
#define R10(a,b,c,d,e,f,g,h,i,j) a,b,c,d,e,f,g,h,i,j
#define R40(m) \
    R10( m, 0,-m, 1, 0,-1, m,-m, 0, 1), \
    R10( 0, m,-m, 0, 1,-1, 0, m, 1,-m), \
    R10(-m, 0, 1, m,-1, 0, 1, 0,-m, m), \
    R10( 1,-m, 0, 0, m, 1,-1, m, 0,-1)
#define R400(m) { \
    R40(m), R40(-m), R40(m), R40(-m), R40(m), \
    R40(-m), R40(m), R40(-m), R40(m), R40(-m)  \
}

R400( 1),
R400(-1),
R400( 2),
R400(-2)

#undef R400
#undef R40
#undef R10
