// Conv2: 16 output channels, 8 input channels, 3x3 kernel
#define W3(a,b,c, d,e,f, g,h,i) { {a,b,c}, {d,e,f}, {g,h,i} }
#define W8(x) { \
    W3( x, 0,-x,  0, x, 0, -x, 0, x), \
    W3( 0,-x, x,  x, 0,-x,  0, x, 0), \
    W3(-x, x, 0,  0,-x, x,  x, 0,-x), \
    W3( x, x,-x, -x, 0, x,  0,-x, x), \
    W3( 0, x, 0, -x, 0, x,  x,-x, 0), \
    W3( x, 0,-x,  0, x, x, -x, 0,-x), \
    W3(-x,-x, x,  x, 0, 0,  0, x,-x), \
    W3( 0, x, x, -x,-x, 0,  x, 0, x)  \
}
W8( 1), W8(-1), W8( 2), W8(-2),
W8( 1), W8( 1), W8(-1), W8( 2),
W8(-1), W8( 2), W8(-2), W8( 1),
W8( 2), W8(-1), W8( 1), W8(-2)
#undef W8
#undef W3
