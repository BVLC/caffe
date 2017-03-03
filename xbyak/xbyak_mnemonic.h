const char *getVersionString() const { return "4.87"; }
void packssdw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x6B); }
void packsswb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x63); }
void packuswb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x67); }
void pand(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xDB); }
void pandn(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xDF); }
void pmaddwd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF5); }
void pmulhuw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE4); }
void pmulhw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE5); }
void pmullw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xD5); }
void por(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xEB); }
void punpckhbw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x68); }
void punpckhwd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x69); }
void punpckhdq(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x6A); }
void punpcklbw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x60); }
void punpcklwd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x61); }
void punpckldq(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x62); }
void pxor(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xEF); }
void pavgb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE0); }
void pavgw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE3); }
void pmaxsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xEE); }
void pmaxub(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xDE); }
void pminsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xEA); }
void pminub(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xDA); }
void psadbw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF6); }
void paddq(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xD4); }
void pmuludq(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF4); }
void psubq(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xFB); }
void paddb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xFC); }
void paddw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xFD); }
void paddd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xFE); }
void paddsb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xEC); }
void paddsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xED); }
void paddusb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xDC); }
void paddusw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xDD); }
void pcmpeqb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x74); }
void pcmpeqw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x75); }
void pcmpeqd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x76); }
void pcmpgtb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x64); }
void pcmpgtw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x65); }
void pcmpgtd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x66); }
void psllw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF1); }
void pslld(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF2); }
void psllq(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF3); }
void psraw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE1); }
void psrad(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE2); }
void psrlw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xD1); }
void psrld(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xD2); }
void psrlq(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xD3); }
void psubb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF8); }
void psubw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xF9); }
void psubd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xFA); }
void psubsb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE8); }
void psubsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xE9); }
void psubusb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xD8); }
void psubusw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0xD9); }
void psllw(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x71, 6); }
void pslld(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x72, 6); }
void psllq(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x73, 6); }
void psraw(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x71, 4); }
void psrad(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x72, 4); }
void psrlw(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x71, 2); }
void psrld(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x72, 2); }
void psrlq(const Mmx& mmx, int imm8) { opMMX_IMM(mmx, imm8, 0x73, 2); }
void pslldq(const Xmm& xmm, int imm8) { opMMX_IMM(xmm, imm8, 0x73, 7); }
void psrldq(const Xmm& xmm, int imm8) { opMMX_IMM(xmm, imm8, 0x73, 3); }
void pshufw(const Mmx& mmx, const Operand& op, uint8 imm8) { opMMX(mmx, op, 0x70, 0x00, imm8); }
void pshuflw(const Mmx& mmx, const Operand& op, uint8 imm8) { opMMX(mmx, op, 0x70, 0xF2, imm8); }
void pshufhw(const Mmx& mmx, const Operand& op, uint8 imm8) { opMMX(mmx, op, 0x70, 0xF3, imm8); }
void pshufd(const Mmx& mmx, const Operand& op, uint8 imm8) { opMMX(mmx, op, 0x70, 0x66, imm8); }
void movdqa(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x6F, 0x66); }
void movdqa(const Address& addr, const Xmm& xmm) { db(0x66); opModM(addr, xmm, 0x0F, 0x7F); }
void movdqu(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x6F, 0xF3); }
void movdqu(const Address& addr, const Xmm& xmm) { db(0xF3); opModM(addr, xmm, 0x0F, 0x7F); }
void movaps(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x28, 0x100); }
void movaps(const Address& addr, const Xmm& xmm) { opModM(addr, xmm, 0x0F, 0x29); }
void movss(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x10, 0xF3); }
void movss(const Address& addr, const Xmm& xmm) { db(0xF3); opModM(addr, xmm, 0x0F, 0x11); }
void movups(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x10, 0x100); }
void movups(const Address& addr, const Xmm& xmm) { opModM(addr, xmm, 0x0F, 0x11); }
void movapd(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x28, 0x66); }
void movapd(const Address& addr, const Xmm& xmm) { db(0x66); opModM(addr, xmm, 0x0F, 0x29); }
void movsd(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x10, 0xF2); }
void movsd(const Address& addr, const Xmm& xmm) { db(0xF2); opModM(addr, xmm, 0x0F, 0x11); }
void movupd(const Xmm& xmm, const Operand& op) { opMMX(xmm, op, 0x10, 0x66); }
void movupd(const Address& addr, const Xmm& xmm) { db(0x66); opModM(addr, xmm, 0x0F, 0x11); }
void addps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x58, 0x100, isXMM_XMMorMEM); }
void addss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x58, 0xF3, isXMM_XMMorMEM); }
void addpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x58, 0x66, isXMM_XMMorMEM); }
void addsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x58, 0xF2, isXMM_XMMorMEM); }
void andnps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x55, 0x100, isXMM_XMMorMEM); }
void andnpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x55, 0x66, isXMM_XMMorMEM); }
void andps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x54, 0x100, isXMM_XMMorMEM); }
void andpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x54, 0x66, isXMM_XMMorMEM); }
void cmpps(const Xmm& xmm, const Operand& op, uint8 imm8) { opGen(xmm, op, 0xC2, 0x100, isXMM_XMMorMEM, imm8); }
void cmpss(const Xmm& xmm, const Operand& op, uint8 imm8) { opGen(xmm, op, 0xC2, 0xF3, isXMM_XMMorMEM, imm8); }
void cmppd(const Xmm& xmm, const Operand& op, uint8 imm8) { opGen(xmm, op, 0xC2, 0x66, isXMM_XMMorMEM, imm8); }
void cmpsd(const Xmm& xmm, const Operand& op, uint8 imm8) { opGen(xmm, op, 0xC2, 0xF2, isXMM_XMMorMEM, imm8); }
void divps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5E, 0x100, isXMM_XMMorMEM); }
void divss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5E, 0xF3, isXMM_XMMorMEM); }
void divpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5E, 0x66, isXMM_XMMorMEM); }
void divsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5E, 0xF2, isXMM_XMMorMEM); }
void maxps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5F, 0x100, isXMM_XMMorMEM); }
void maxss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5F, 0xF3, isXMM_XMMorMEM); }
void maxpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5F, 0x66, isXMM_XMMorMEM); }
void maxsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5F, 0xF2, isXMM_XMMorMEM); }
void minps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5D, 0x100, isXMM_XMMorMEM); }
void minss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5D, 0xF3, isXMM_XMMorMEM); }
void minpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5D, 0x66, isXMM_XMMorMEM); }
void minsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5D, 0xF2, isXMM_XMMorMEM); }
void mulps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x59, 0x100, isXMM_XMMorMEM); }
void mulss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x59, 0xF3, isXMM_XMMorMEM); }
void mulpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x59, 0x66, isXMM_XMMorMEM); }
void mulsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x59, 0xF2, isXMM_XMMorMEM); }
void orps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x56, 0x100, isXMM_XMMorMEM); }
void orpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x56, 0x66, isXMM_XMMorMEM); }
void rcpps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x53, 0x100, isXMM_XMMorMEM); }
void rcpss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x53, 0xF3, isXMM_XMMorMEM); }
void rsqrtps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x52, 0x100, isXMM_XMMorMEM); }
void rsqrtss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x52, 0xF3, isXMM_XMMorMEM); }
void shufps(const Xmm& xmm, const Operand& op, uint8 imm8) { opGen(xmm, op, 0xC6, 0x100, isXMM_XMMorMEM, imm8); }
void shufpd(const Xmm& xmm, const Operand& op, uint8 imm8) { opGen(xmm, op, 0xC6, 0x66, isXMM_XMMorMEM, imm8); }
void sqrtps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x51, 0x100, isXMM_XMMorMEM); }
void sqrtss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x51, 0xF3, isXMM_XMMorMEM); }
void sqrtpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x51, 0x66, isXMM_XMMorMEM); }
void sqrtsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x51, 0xF2, isXMM_XMMorMEM); }
void subps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5C, 0x100, isXMM_XMMorMEM); }
void subss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5C, 0xF3, isXMM_XMMorMEM); }
void subpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5C, 0x66, isXMM_XMMorMEM); }
void subsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5C, 0xF2, isXMM_XMMorMEM); }
void unpckhps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x15, 0x100, isXMM_XMMorMEM); }
void unpckhpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x15, 0x66, isXMM_XMMorMEM); }
void unpcklps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x14, 0x100, isXMM_XMMorMEM); }
void unpcklpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x14, 0x66, isXMM_XMMorMEM); }
void xorps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x57, 0x100, isXMM_XMMorMEM); }
void xorpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x57, 0x66, isXMM_XMMorMEM); }
void maskmovdqu(const Xmm& reg1, const Xmm& reg2) { db(0x66);  opModR(reg1, reg2, 0x0F, 0xF7); }
void movhlps(const Xmm& reg1, const Xmm& reg2) {  opModR(reg1, reg2, 0x0F, 0x12); }
void movlhps(const Xmm& reg1, const Xmm& reg2) {  opModR(reg1, reg2, 0x0F, 0x16); }
void punpckhqdq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x6D, 0x66, isXMM_XMMorMEM); }
void punpcklqdq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x6C, 0x66, isXMM_XMMorMEM); }
void comiss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x2F, 0x100, isXMM_XMMorMEM); }
void ucomiss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x2E, 0x100, isXMM_XMMorMEM); }
void comisd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x2F, 0x66, isXMM_XMMorMEM); }
void ucomisd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x2E, 0x66, isXMM_XMMorMEM); }
void cvtpd2ps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5A, 0x66, isXMM_XMMorMEM); }
void cvtps2pd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5A, 0x100, isXMM_XMMorMEM); }
void cvtsd2ss(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5A, 0xF2, isXMM_XMMorMEM); }
void cvtss2sd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5A, 0xF3, isXMM_XMMorMEM); }
void cvtpd2dq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xE6, 0xF2, isXMM_XMMorMEM); }
void cvttpd2dq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xE6, 0x66, isXMM_XMMorMEM); }
void cvtdq2pd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xE6, 0xF3, isXMM_XMMorMEM); }
void cvtps2dq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5B, 0x66, isXMM_XMMorMEM); }
void cvttps2dq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5B, 0xF3, isXMM_XMMorMEM); }
void cvtdq2ps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x5B, 0x100, isXMM_XMMorMEM); }
void addsubpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xD0, 0x66, isXMM_XMMorMEM); }
void addsubps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xD0, 0xF2, isXMM_XMMorMEM); }
void haddpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x7C, 0x66, isXMM_XMMorMEM); }
void haddps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x7C, 0xF2, isXMM_XMMorMEM); }
void hsubpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x7D, 0x66, isXMM_XMMorMEM); }
void hsubps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x7D, 0xF2, isXMM_XMMorMEM); }
void movddup(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x12, 0xF2, isXMM_XMMorMEM); }
void movshdup(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x16, 0xF3, isXMM_XMMorMEM); }
void movsldup(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x12, 0xF3, isXMM_XMMorMEM); }
void cvtpi2ps(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2A, 0x100, isXMM_MMXorMEM); }
void cvtps2pi(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2D, 0x100, isMMX_XMMorMEM); }
void cvtsi2ss(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2A, 0xF3, isXMM_REG32orMEM); }
void cvtss2si(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2D, 0xF3, isREG32_XMMorMEM); }
void cvttps2pi(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2C, 0x100, isMMX_XMMorMEM); }
void cvttss2si(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2C, 0xF3, isREG32_XMMorMEM); }
void cvtpi2pd(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2A, 0x66, isXMM_MMXorMEM); }
void cvtpd2pi(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2D, 0x66, isMMX_XMMorMEM); }
void cvtsi2sd(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2A, 0xF2, isXMM_REG32orMEM); }
void cvtsd2si(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2D, 0xF2, isREG32_XMMorMEM); }
void cvttpd2pi(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2C, 0x66, isMMX_XMMorMEM); }
void cvttsd2si(const Operand& reg, const Operand& op) { opGen(reg, op, 0x2C, 0xF2, isREG32_XMMorMEM); }
void prefetcht0(const Address& addr) { opModM(addr, Reg32(1), 0x0F, B00011000); }
void prefetcht1(const Address& addr) { opModM(addr, Reg32(2), 0x0F, B00011000); }
void prefetcht2(const Address& addr) { opModM(addr, Reg32(3), 0x0F, B00011000); }
void prefetchnta(const Address& addr) { opModM(addr, Reg32(0), 0x0F, B00011000); }
void movhps(const Operand& op1, const Operand& op2) { opMovXMM(op1, op2, 0x16, 0x100); }
void movlps(const Operand& op1, const Operand& op2) { opMovXMM(op1, op2, 0x12, 0x100); }
void movhpd(const Operand& op1, const Operand& op2) { opMovXMM(op1, op2, 0x16, 0x66); }
void movlpd(const Operand& op1, const Operand& op2) { opMovXMM(op1, op2, 0x12, 0x66); }
void cmovo(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 0); }
void jo(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x70, 0x80, 0x0F); }
void jo(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x70, 0x80, 0x0F); }
void seto(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 0); }
void cmovno(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 1); }
void jno(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x71, 0x81, 0x0F); }
void jno(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x71, 0x81, 0x0F); }
void setno(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 1); }
void cmovb(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 2); }
void jb(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x72, 0x82, 0x0F); }
void jb(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x72, 0x82, 0x0F); }
void setb(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 2); }
void cmovc(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 2); }
void jc(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x72, 0x82, 0x0F); }
void jc(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x72, 0x82, 0x0F); }
void setc(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 2); }
void cmovnae(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 2); }
void jnae(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x72, 0x82, 0x0F); }
void jnae(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x72, 0x82, 0x0F); }
void setnae(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 2); }
void cmovnb(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 3); }
void jnb(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x73, 0x83, 0x0F); }
void jnb(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x73, 0x83, 0x0F); }
void setnb(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 3); }
void cmovae(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 3); }
void jae(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x73, 0x83, 0x0F); }
void jae(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x73, 0x83, 0x0F); }
void setae(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 3); }
void cmovnc(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 3); }
void jnc(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x73, 0x83, 0x0F); }
void jnc(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x73, 0x83, 0x0F); }
void setnc(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 3); }
void cmove(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 4); }
void je(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x74, 0x84, 0x0F); }
void je(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x74, 0x84, 0x0F); }
void sete(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 4); }
void cmovz(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 4); }
void jz(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x74, 0x84, 0x0F); }
void jz(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x74, 0x84, 0x0F); }
void setz(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 4); }
void cmovne(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 5); }
void jne(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x75, 0x85, 0x0F); }
void jne(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x75, 0x85, 0x0F); }
void setne(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 5); }
void cmovnz(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 5); }
void jnz(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x75, 0x85, 0x0F); }
void jnz(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x75, 0x85, 0x0F); }
void setnz(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 5); }
void cmovbe(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 6); }
void jbe(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x76, 0x86, 0x0F); }
void jbe(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x76, 0x86, 0x0F); }
void setbe(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 6); }
void cmovna(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 6); }
void jna(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x76, 0x86, 0x0F); }
void jna(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x76, 0x86, 0x0F); }
void setna(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 6); }
void cmovnbe(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 7); }
void jnbe(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x77, 0x87, 0x0F); }
void jnbe(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x77, 0x87, 0x0F); }
void setnbe(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 7); }
void cmova(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 7); }
void ja(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x77, 0x87, 0x0F); }
void ja(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x77, 0x87, 0x0F); }
void seta(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 7); }
void cmovs(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 8); }
void js(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x78, 0x88, 0x0F); }
void js(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x78, 0x88, 0x0F); }
void sets(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 8); }
void cmovns(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 9); }
void jns(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x79, 0x89, 0x0F); }
void jns(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x79, 0x89, 0x0F); }
void setns(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 9); }
void cmovp(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 10); }
void jp(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7A, 0x8A, 0x0F); }
void jp(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7A, 0x8A, 0x0F); }
void setp(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 10); }
void cmovpe(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 10); }
void jpe(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7A, 0x8A, 0x0F); }
void jpe(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7A, 0x8A, 0x0F); }
void setpe(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 10); }
void cmovnp(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 11); }
void jnp(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7B, 0x8B, 0x0F); }
void jnp(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7B, 0x8B, 0x0F); }
void setnp(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 11); }
void cmovpo(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 11); }
void jpo(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7B, 0x8B, 0x0F); }
void jpo(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7B, 0x8B, 0x0F); }
void setpo(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 11); }
void cmovl(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 12); }
void jl(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7C, 0x8C, 0x0F); }
void jl(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7C, 0x8C, 0x0F); }
void setl(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 12); }
void cmovnge(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 12); }
void jnge(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7C, 0x8C, 0x0F); }
void jnge(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7C, 0x8C, 0x0F); }
void setnge(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 12); }
void cmovnl(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 13); }
void jnl(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7D, 0x8D, 0x0F); }
void jnl(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7D, 0x8D, 0x0F); }
void setnl(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 13); }
void cmovge(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 13); }
void jge(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7D, 0x8D, 0x0F); }
void jge(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7D, 0x8D, 0x0F); }
void setge(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 13); }
void cmovle(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 14); }
void jle(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7E, 0x8E, 0x0F); }
void jle(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7E, 0x8E, 0x0F); }
void setle(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 14); }
void cmovng(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 14); }
void jng(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7E, 0x8E, 0x0F); }
void jng(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7E, 0x8E, 0x0F); }
void setng(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 14); }
void cmovnle(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 15); }
void jnle(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7F, 0x8F, 0x0F); }
void jnle(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7F, 0x8F, 0x0F); }
void setnle(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 15); }
void cmovg(const Reg32e& reg, const Operand& op) { opModRM(reg, op, op.isREG(i32e), op.isMEM(), 0x0F, B01000000 | 15); }
void jg(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0x7F, 0x8F, 0x0F); }
void jg(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0x7F, 0x8F, 0x0F); }
void setg(const Operand& op) { opR_ModM(op, 8, 0, 0x0F, B10010000 | 15); }
#ifdef XBYAK32
void jcxz(std::string label) { db(0x67); opJmp(label, T_SHORT, 0xe3, 0, 0); }
void jcxz(const Label& label) { db(0x67); opJmp(label, T_SHORT, 0xe3, 0, 0); }
void jecxz(std::string label) { opJmp(label, T_SHORT, 0xe3, 0, 0); }
void jecxz(const Label& label) { opJmp(label, T_SHORT, 0xe3, 0, 0); }
#else
void jecxz(std::string label) { db(0x67); opJmp(label, T_SHORT, 0xe3, 0, 0); }
void jecxz(const Label& label) { db(0x67); opJmp(label, T_SHORT, 0xe3, 0, 0); }
void jrcxz(std::string label) { opJmp(label, T_SHORT, 0xe3, 0, 0); }
void jrcxz(const Label& label) { opJmp(label, T_SHORT, 0xe3, 0, 0); }
#endif
#ifdef XBYAK64
void cdqe() { db(0x48); db(0x98); }
void cqo() { db(0x48); db(0x99); }
#else
void aaa() { db(0x37); }
void aad() { db(0xD5); db(0x0A); }
void aam() { db(0xD4); db(0x0A); }
void aas() { db(0x3F); }
void daa() { db(0x27); }
void das() { db(0x2F); }
void popad() { db(0x61); }
void popfd() { db(0x9D); }
void pusha() { db(0x60); }
void pushad() { db(0x60); }
void pushfd() { db(0x9C); }
void popa() { db(0x61); }
#endif
void cbw() { db(0x66); db(0x98); }
void cdq() { db(0x99); }
void clc() { db(0xF8); }
void cld() { db(0xFC); }
void cli() { db(0xFA); }
void cmc() { db(0xF5); }
void cpuid() { db(0x0F); db(0xA2); }
void cwd() { db(0x66); db(0x99); }
void cwde() { db(0x98); }
void lahf() { db(0x9F); }
void lock() { db(0xF0); }
void nop() { db(0x90); }
void sahf() { db(0x9E); }
void stc() { db(0xF9); }
void std() { db(0xFD); }
void sti() { db(0xFB); }
void emms() { db(0x0F); db(0x77); }
void pause() { db(0xF3); db(0x90); }
void sfence() { db(0x0F); db(0xAE); db(0xF8); }
void lfence() { db(0x0F); db(0xAE); db(0xE8); }
void mfence() { db(0x0F); db(0xAE); db(0xF0); }
void monitor() { db(0x0F); db(0x01); db(0xC8); }
void mwait() { db(0x0F); db(0x01); db(0xC9); }
void rdmsr() { db(0x0F); db(0x32); }
void rdpmc() { db(0x0F); db(0x33); }
void rdtsc() { db(0x0F); db(0x31); }
void rdtscp() { db(0x0F); db(0x01); db(0xF9); }
void ud2() { db(0x0F); db(0x0B); }
void wait() { db(0x9B); }
void fwait() { db(0x9B); }
void wbinvd() { db(0x0F); db(0x09); }
void wrmsr() { db(0x0F); db(0x30); }
void xlatb() { db(0xD7); }
void popf() { db(0x9D); }
void pushf() { db(0x9C); }
void stac() { db(0x0F); db(0x01); db(0xCB); }
void vzeroall() { db(0xC5); db(0xFC); db(0x77); }
void vzeroupper() { db(0xC5); db(0xF8); db(0x77); }
void xgetbv() { db(0x0F); db(0x01); db(0xD0); }
void f2xm1() { db(0xD9); db(0xF0); }
void fabs() { db(0xD9); db(0xE1); }
void faddp() { db(0xDE); db(0xC1); }
void fchs() { db(0xD9); db(0xE0); }
void fcom() { db(0xD8); db(0xD1); }
void fcomp() { db(0xD8); db(0xD9); }
void fcompp() { db(0xDE); db(0xD9); }
void fcos() { db(0xD9); db(0xFF); }
void fdecstp() { db(0xD9); db(0xF6); }
void fdivp() { db(0xDE); db(0xF9); }
void fdivrp() { db(0xDE); db(0xF1); }
void fincstp() { db(0xD9); db(0xF7); }
void finit() { db(0x9B); db(0xDB); db(0xE3); }
void fninit() { db(0xDB); db(0xE3); }
void fld1() { db(0xD9); db(0xE8); }
void fldl2t() { db(0xD9); db(0xE9); }
void fldl2e() { db(0xD9); db(0xEA); }
void fldpi() { db(0xD9); db(0xEB); }
void fldlg2() { db(0xD9); db(0xEC); }
void fldln2() { db(0xD9); db(0xED); }
void fldz() { db(0xD9); db(0xEE); }
void fmulp() { db(0xDE); db(0xC9); }
void fnop() { db(0xD9); db(0xD0); }
void fpatan() { db(0xD9); db(0xF3); }
void fprem() { db(0xD9); db(0xF8); }
void fprem1() { db(0xD9); db(0xF5); }
void fptan() { db(0xD9); db(0xF2); }
void frndint() { db(0xD9); db(0xFC); }
void fscale() { db(0xD9); db(0xFD); }
void fsin() { db(0xD9); db(0xFE); }
void fsincos() { db(0xD9); db(0xFB); }
void fsqrt() { db(0xD9); db(0xFA); }
void fsubp() { db(0xDE); db(0xE9); }
void fsubrp() { db(0xDE); db(0xE1); }
void ftst() { db(0xD9); db(0xE4); }
void fucom() { db(0xDD); db(0xE1); }
void fucomp() { db(0xDD); db(0xE9); }
void fucompp() { db(0xDA); db(0xE9); }
void fxam() { db(0xD9); db(0xE5); }
void fxch() { db(0xD9); db(0xC9); }
void fxtract() { db(0xD9); db(0xF4); }
void fyl2x() { db(0xD9); db(0xF1); }
void fyl2xp1() { db(0xD9); db(0xF9); }
void adc(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x10); }
void adc(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x10, 2); }
void add(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x00); }
void add(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x00, 0); }
void and_(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x20); }
void and_(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x20, 4); }
#ifndef XBYAK_NO_OP_NAMES
void and(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x20); }
void and(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x20, 4); }
#endif
void cmp(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x38); }
void cmp(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x38, 7); }
void or_(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x08); }
void or_(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x08, 1); }
#ifndef XBYAK_NO_OP_NAMES
void or(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x08); }
void or(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x08, 1); }
#endif
void sbb(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x18); }
void sbb(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x18, 3); }
void sub(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x28); }
void sub(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x28, 5); }
void xor_(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x30); }
void xor_(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x30, 6); }
#ifndef XBYAK_NO_OP_NAMES
void xor(const Operand& op1, const Operand& op2) { opRM_RM(op1, op2, 0x30); }
void xor(const Operand& op, uint32 imm) { opRM_I(op, imm, 0x30, 6); }
#endif
void dec(const Operand& op) { opIncDec(op, 0x48, 1); }
void inc(const Operand& op) { opIncDec(op, 0x40, 0); }
void bt(const Operand& op, const Reg& reg) { opModRM(reg, op, op.isREG(16|32|64) && op.getBit() == reg.getBit(), op.isMEM(), 0x0f, 0xa3); }
void bt(const Operand& op, uint8 imm) { opR_ModM(op, 16|32|64, 4, 0x0f, 0xba, NONE, false, 1); db(imm); }
void bts(const Operand& op, const Reg& reg) { opModRM(reg, op, op.isREG(16|32|64) && op.getBit() == reg.getBit(), op.isMEM(), 0x0f, 0xab); }
void bts(const Operand& op, uint8 imm) { opR_ModM(op, 16|32|64, 5, 0x0f, 0xba, NONE, false, 1); db(imm); }
void btr(const Operand& op, const Reg& reg) { opModRM(reg, op, op.isREG(16|32|64) && op.getBit() == reg.getBit(), op.isMEM(), 0x0f, 0xb3); }
void btr(const Operand& op, uint8 imm) { opR_ModM(op, 16|32|64, 6, 0x0f, 0xba, NONE, false, 1); db(imm); }
void btc(const Operand& op, const Reg& reg) { opModRM(reg, op, op.isREG(16|32|64) && op.getBit() == reg.getBit(), op.isMEM(), 0x0f, 0xbb); }
void btc(const Operand& op, uint8 imm) { opR_ModM(op, 16|32|64, 7, 0x0f, 0xba, NONE, false, 1); db(imm); }
void div(const Operand& op) { opR_ModM(op, 0, 6, 0xF6); }
void idiv(const Operand& op) { opR_ModM(op, 0, 7, 0xF6); }
void imul(const Operand& op) { opR_ModM(op, 0, 5, 0xF6); }
void mul(const Operand& op) { opR_ModM(op, 0, 4, 0xF6); }
void neg(const Operand& op) { opR_ModM(op, 0, 3, 0xF6); }
void not_(const Operand& op) { opR_ModM(op, 0, 2, 0xF6); }
#ifndef XBYAK_NO_OP_NAMES
void not(const Operand& op) { opR_ModM(op, 0, 2, 0xF6); }
#endif
void rcl(const Operand& op, int imm) { opShift(op, imm, 2); }
void rcl(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 2); }
void rcr(const Operand& op, int imm) { opShift(op, imm, 3); }
void rcr(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 3); }
void rol(const Operand& op, int imm) { opShift(op, imm, 0); }
void rol(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 0); }
void ror(const Operand& op, int imm) { opShift(op, imm, 1); }
void ror(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 1); }
void sar(const Operand& op, int imm) { opShift(op, imm, 7); }
void sar(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 7); }
void shl(const Operand& op, int imm) { opShift(op, imm, 4); }
void shl(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 4); }
void shr(const Operand& op, int imm) { opShift(op, imm, 5); }
void shr(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 5); }
void sal(const Operand& op, int imm) { opShift(op, imm, 4); }
void sal(const Operand& op, const Reg8& _cl) { opShift(op, _cl, 4); }
void shld(const Operand& op, const Reg& reg, uint8 imm) { opShxd(op, reg, imm, 0xA4); }
void shld(const Operand& op, const Reg& reg, const Reg8& _cl) { opShxd(op, reg, 0, 0xA4, &_cl); }
void shrd(const Operand& op, const Reg& reg, uint8 imm) { opShxd(op, reg, imm, 0xAC); }
void shrd(const Operand& op, const Reg& reg, const Reg8& _cl) { opShxd(op, reg, 0, 0xAC, &_cl); }
void bsf(const Reg&reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0xBC); }
void bsr(const Reg&reg, const Operand& op) { opModRM(reg, op, op.isREG(16 | i32e), op.isMEM(), 0x0F, 0xBD); }
void popcnt(const Reg&reg, const Operand& op) { opSp1(reg, op, 0xF3, 0x0F, 0xB8); }
void tzcnt(const Reg&reg, const Operand& op) { opSp1(reg, op, 0xF3, 0x0F, 0xBC); }
void lzcnt(const Reg&reg, const Operand& op) { opSp1(reg, op, 0xF3, 0x0F, 0xBD); }
void pshufb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x00, 0x66, NONE, 0x38); }
void phaddw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x01, 0x66, NONE, 0x38); }
void phaddd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x02, 0x66, NONE, 0x38); }
void phaddsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x03, 0x66, NONE, 0x38); }
void pmaddubsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x04, 0x66, NONE, 0x38); }
void phsubw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x05, 0x66, NONE, 0x38); }
void phsubd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x06, 0x66, NONE, 0x38); }
void phsubsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x07, 0x66, NONE, 0x38); }
void psignb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x08, 0x66, NONE, 0x38); }
void psignw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x09, 0x66, NONE, 0x38); }
void psignd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x0A, 0x66, NONE, 0x38); }
void pmulhrsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x0B, 0x66, NONE, 0x38); }
void pabsb(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x1C, 0x66, NONE, 0x38); }
void pabsw(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x1D, 0x66, NONE, 0x38); }
void pabsd(const Mmx& mmx, const Operand& op) { opMMX(mmx, op, 0x1E, 0x66, NONE, 0x38); }
void palignr(const Mmx& mmx, const Operand& op, int imm) { opMMX(mmx, op, 0x0f, 0x66, static_cast<uint8>(imm), 0x3a); }
void blendvpd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x15, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void blendvps(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x14, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void packusdw(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x2B, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pblendvb(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x10, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pcmpeqq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x29, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void ptest(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x17, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovsxbw(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x20, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovsxbd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x21, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovsxbq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x22, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovsxwd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x23, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovsxwq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x24, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovsxdq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x25, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovzxbw(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x30, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovzxbd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x31, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovzxbq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x32, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovzxwd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x33, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovzxwq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x34, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmovzxdq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x35, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pminsb(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x38, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pminsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x39, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pminuw(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x3A, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pminud(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x3B, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmaxsb(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x3C, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmaxsd(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x3D, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmaxuw(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x3E, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmaxud(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x3F, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmuldq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x28, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pmulld(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x40, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void phminposuw(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x41, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void pcmpgtq(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0x37, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void aesdec(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xDE, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void aesdeclast(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xDF, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void aesenc(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xDC, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void aesenclast(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xDD, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void aesimc(const Xmm& xmm, const Operand& op) { opGen(xmm, op, 0xDB, 0x66, isXMM_XMMorMEM, NONE, 0x38); }
void blendpd(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x0D, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void blendps(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x0C, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void dppd(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x41, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void dpps(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x40, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void mpsadbw(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x42, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void pblendw(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x0E, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void roundps(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x08, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void roundpd(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x09, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void roundss(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x0A, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void roundsd(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x0B, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void pcmpestrm(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x60, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void pcmpestri(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x61, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void pcmpistrm(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x62, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void pcmpistri(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x63, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void pclmulqdq(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0x44, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void aeskeygenassist(const Xmm& xmm, const Operand& op, int imm) { opGen(xmm, op, 0xDF, 0x66, isXMM_XMMorMEM, static_cast<uint8>(imm), 0x3A); }
void pclmullqlqdq(const Xmm& xmm, const Operand& op) { pclmulqdq(xmm, op, 0x00); }
void pclmulhqlqdq(const Xmm& xmm, const Operand& op) { pclmulqdq(xmm, op, 0x01); }
void pclmullqhdq(const Xmm& xmm, const Operand& op) { pclmulqdq(xmm, op, 0x10); }
void pclmulhqhdq(const Xmm& xmm, const Operand& op) { pclmulqdq(xmm, op, 0x11); }
void ldmxcsr(const Address& addr) { opModM(addr, Reg32(2), 0x0F, 0xAE); }
void stmxcsr(const Address& addr) { opModM(addr, Reg32(3), 0x0F, 0xAE); }
void clflush(const Address& addr) { opModM(addr, Reg32(7), 0x0F, 0xAE); }
void fldcw(const Address& addr) { opModM(addr, Reg32(5), 0xD9, 0x100); }
void fstcw(const Address& addr) { db(0x9B); opModM(addr, Reg32(7), 0xD9, NONE); }
void movntpd(const Address& addr, const Xmm& reg) { opModM(addr, Reg16(reg.getIdx()), 0x0F, 0x2B); }
void movntdq(const Address& addr, const Xmm& reg) { opModM(addr, Reg16(reg.getIdx()), 0x0F, 0xE7); }
void movsx(const Reg& reg, const Operand& op) { opMovxx(reg, op, 0xBE); }
void movzx(const Reg& reg, const Operand& op) { opMovxx(reg, op, 0xB6); }
void fadd(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 0, 0); }
void fiadd(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 0, 0); }
void fcom(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 2, 0); }
void fcomp(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 3, 0); }
void fdiv(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 6, 0); }
void fidiv(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 6, 0); }
void fdivr(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 7, 0); }
void fidivr(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 7, 0); }
void ficom(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 2, 0); }
void ficomp(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 3, 0); }
void fild(const Address& addr) { opFpuMem(addr, 0xDF, 0xDB, 0xDF, 0, 5); }
void fist(const Address& addr) { opFpuMem(addr, 0xDF, 0xDB, 0x00, 2, 0); }
void fistp(const Address& addr) { opFpuMem(addr, 0xDF, 0xDB, 0xDF, 3, 7); }
void fisttp(const Address& addr) { opFpuMem(addr, 0xDF, 0xDB, 0xDD, 1, 0); }
void fld(const Address& addr) { opFpuMem(addr, 0x00, 0xD9, 0xDD, 0, 0); }
void fmul(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 1, 0); }
void fimul(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 1, 0); }
void fst(const Address& addr) { opFpuMem(addr, 0x00, 0xD9, 0xDD, 2, 0); }
void fstp(const Address& addr) { opFpuMem(addr, 0x00, 0xD9, 0xDD, 3, 0); }
void fsub(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 4, 0); }
void fisub(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 4, 0); }
void fsubr(const Address& addr) { opFpuMem(addr, 0x00, 0xD8, 0xDC, 5, 0); }
void fisubr(const Address& addr) { opFpuMem(addr, 0xDE, 0xDA, 0x00, 5, 0); }
void fadd(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xD8C0, 0xDCC0); }
void fadd(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xD8C0, 0xDCC0); }
void faddp(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0x0000, 0xDEC0); }
void faddp(const Fpu& reg1) { opFpuFpu(reg1, st0, 0x0000, 0xDEC0); }
void fcmovb(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDAC0, 0x00C0); }
void fcmovb(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDAC0, 0x00C0); }
void fcmove(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDAC8, 0x00C8); }
void fcmove(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDAC8, 0x00C8); }
void fcmovbe(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDAD0, 0x00D0); }
void fcmovbe(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDAD0, 0x00D0); }
void fcmovu(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDAD8, 0x00D8); }
void fcmovu(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDAD8, 0x00D8); }
void fcmovnb(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDBC0, 0x00C0); }
void fcmovnb(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDBC0, 0x00C0); }
void fcmovne(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDBC8, 0x00C8); }
void fcmovne(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDBC8, 0x00C8); }
void fcmovnbe(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDBD0, 0x00D0); }
void fcmovnbe(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDBD0, 0x00D0); }
void fcmovnu(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDBD8, 0x00D8); }
void fcmovnu(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDBD8, 0x00D8); }
void fcomi(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDBF0, 0x00F0); }
void fcomi(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDBF0, 0x00F0); }
void fcomip(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDFF0, 0x00F0); }
void fcomip(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDFF0, 0x00F0); }
void fucomi(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDBE8, 0x00E8); }
void fucomi(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDBE8, 0x00E8); }
void fucomip(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xDFE8, 0x00E8); }
void fucomip(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xDFE8, 0x00E8); }
void fdiv(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xD8F0, 0xDCF8); }
void fdiv(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xD8F0, 0xDCF8); }
void fdivp(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0x0000, 0xDEF8); }
void fdivp(const Fpu& reg1) { opFpuFpu(reg1, st0, 0x0000, 0xDEF8); }
void fdivr(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xD8F8, 0xDCF0); }
void fdivr(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xD8F8, 0xDCF0); }
void fdivrp(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0x0000, 0xDEF0); }
void fdivrp(const Fpu& reg1) { opFpuFpu(reg1, st0, 0x0000, 0xDEF0); }
void fmul(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xD8C8, 0xDCC8); }
void fmul(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xD8C8, 0xDCC8); }
void fmulp(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0x0000, 0xDEC8); }
void fmulp(const Fpu& reg1) { opFpuFpu(reg1, st0, 0x0000, 0xDEC8); }
void fsub(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xD8E0, 0xDCE8); }
void fsub(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xD8E0, 0xDCE8); }
void fsubp(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0x0000, 0xDEE8); }
void fsubp(const Fpu& reg1) { opFpuFpu(reg1, st0, 0x0000, 0xDEE8); }
void fsubr(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0xD8E8, 0xDCE0); }
void fsubr(const Fpu& reg1) { opFpuFpu(st0, reg1, 0xD8E8, 0xDCE0); }
void fsubrp(const Fpu& reg1, const Fpu& reg2) { opFpuFpu(reg1, reg2, 0x0000, 0xDEE0); }
void fsubrp(const Fpu& reg1) { opFpuFpu(reg1, st0, 0x0000, 0xDEE0); }
void fcom(const Fpu& reg) { opFpu(reg, 0xD8, 0xD0); }
void fcomp(const Fpu& reg) { opFpu(reg, 0xD8, 0xD8); }
void ffree(const Fpu& reg) { opFpu(reg, 0xDD, 0xC0); }
void fld(const Fpu& reg) { opFpu(reg, 0xD9, 0xC0); }
void fst(const Fpu& reg) { opFpu(reg, 0xDD, 0xD0); }
void fstp(const Fpu& reg) { opFpu(reg, 0xDD, 0xD8); }
void fucom(const Fpu& reg) { opFpu(reg, 0xDD, 0xE0); }
void fucomp(const Fpu& reg) { opFpu(reg, 0xDD, 0xE8); }
void fxch(const Fpu& reg) { opFpu(reg, 0xD9, 0xC8); }
void vaddpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_66, 0x58, true); }
void vaddps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F, 0x58, true); }
void vaddsd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F2, 0x58, false); }
void vaddss(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F3, 0x58, false); }
void vsubpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_66, 0x5C, true); }
void vsubps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F, 0x5C, true); }
void vsubsd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F2, 0x5C, false); }
void vsubss(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F3, 0x5C, false); }
void vmulpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_66, 0x59, true); }
void vmulps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F, 0x59, true); }
void vmulsd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F2, 0x59, false); }
void vmulss(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F3, 0x59, false); }
void vdivpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_66, 0x5E, true); }
void vdivps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F, 0x5E, true); }
void vdivsd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F2, 0x5E, false); }
void vdivss(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F3, 0x5E, false); }
void vmaxpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_66, 0x5F, true); }
void vmaxps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F, 0x5F, true); }
void vmaxsd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F2, 0x5F, false); }
void vmaxss(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F3, 0x5F, false); }
void vminpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_66, 0x5D, true); }
void vminps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F, 0x5D, true); }
void vminsd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F2, 0x5D, false); }
void vminss(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F3, 0x5D, false); }
void vandpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_66, 0x54, true); }
void vandps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F, 0x54, true); }
void vandnpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_66, 0x55, true); }
void vandnps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F, 0x55, true); }
void vorpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_66, 0x56, true); }
void vorps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F, 0x56, true); }
void vxorpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_66, 0x57, true); }
void vxorps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F, 0x57, true); }
void vblendpd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x0D, true, 0, imm); }
void vblendpd(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F3A | PP_66, 0x0D, true, 0, imm); }
void vblendps(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x0C, true, 0, imm); }
void vblendps(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F3A | PP_66, 0x0C, true, 0, imm); }
void vdppd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x41, false, 0, imm); }
void vdppd(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F3A | PP_66, 0x41, false, 0, imm); }
void vdpps(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x40, true, 0, imm); }
void vdpps(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F3A | PP_66, 0x40, true, 0, imm); }
void vmpsadbw(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x42, true, 0, imm); }
void vmpsadbw(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F3A | PP_66, 0x42, true, 0, imm); }
void vpblendw(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x0E, true, 0, imm); }
void vpblendw(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F3A | PP_66, 0x0E, true, 0, imm); }
void vpblendd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x02, true, 0, imm); }
void vpblendd(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F3A | PP_66, 0x02, true, 0, imm); }
void vroundsd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x0B, false, 0, imm); }
void vroundsd(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F3A | PP_66, 0x0B, false, 0, imm); }
void vroundss(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x0A, false, 0, imm); }
void vroundss(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F3A | PP_66, 0x0A, false, 0, imm); }
void vpclmulqdq(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x44, false, 0, imm); }
void vpclmulqdq(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F3A | PP_66, 0x44, false, 0, imm); }
void vpermilps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x0C, true, 0); }
void vpermilpd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x0D, true, 0); }
void vpsllvd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x47, true, 0); }
void vpsllvq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x47, true, 1); }
void vpsravd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x46, true, 0); }
void vpsrlvd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x45, true, 0); }
void vpsrlvq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x45, true, 1); }
void vcmppd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xC2, true, -1, imm); }
void vcmppd(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xC2, true, -1, imm); }
void vcmpps(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F, 0xC2, true, -1, imm); }
void vcmpps(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F, 0xC2, true, -1, imm); }
void vcmpsd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_F2, 0xC2, false, -1, imm); }
void vcmpsd(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_F2, 0xC2, false, -1, imm); }
void vcmpss(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_F3, 0xC2, false, -1, imm); }
void vcmpss(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_F3, 0xC2, false, -1, imm); }
void vcvtsd2ss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_F2, 0x5A, false, -1); }
void vcvtsd2ss(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_F2, 0x5A, false, -1); }
void vcvtss2sd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_F3, 0x5A, false, -1); }
void vcvtss2sd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_F3, 0x5A, false, -1); }
void vinsertps(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x21, false, 0, imm); }
void vinsertps(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F3A | PP_66, 0x21, false, 0, imm); }
void vpacksswb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x63, true, -1); }
void vpacksswb(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x63, true, -1); }
void vpackssdw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x6B, true, -1); }
void vpackssdw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x6B, true, -1); }
void vpackuswb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x67, true, -1); }
void vpackuswb(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x67, true, -1); }
void vpackusdw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x2B, true, -1); }
void vpackusdw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x2B, true, -1); }
void vpaddb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xFC, true, -1); }
void vpaddb(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xFC, true, -1); }
void vpaddw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xFD, true, -1); }
void vpaddw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xFD, true, -1); }
void vpaddd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xFE, true, -1); }
void vpaddd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xFE, true, -1); }
void vpaddq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xD4, true, -1); }
void vpaddq(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xD4, true, -1); }
void vpaddsb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xEC, true, -1); }
void vpaddsb(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xEC, true, -1); }
void vpaddsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xED, true, -1); }
void vpaddsw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xED, true, -1); }
void vpaddusb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xDC, true, -1); }
void vpaddusb(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xDC, true, -1); }
void vpaddusw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xDD, true, -1); }
void vpaddusw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xDD, true, -1); }
void vpalignr(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x0F, true, -1, imm); }
void vpalignr(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F3A | PP_66, 0x0F, true, -1, imm); }
void vpand(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xDB, true, -1); }
void vpand(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xDB, true, -1); }
void vpandn(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xDF, true, -1); }
void vpandn(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xDF, true, -1); }
void vpavgb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xE0, true, -1); }
void vpavgb(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xE0, true, -1); }
void vpavgw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xE3, true, -1); }
void vpavgw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xE3, true, -1); }
void vpcmpeqb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x74, true, -1); }
void vpcmpeqb(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x74, true, -1); }
void vpcmpeqw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x75, true, -1); }
void vpcmpeqw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x75, true, -1); }
void vpcmpeqd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x76, true, -1); }
void vpcmpeqd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x76, true, -1); }
void vpcmpeqq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x29, true, -1); }
void vpcmpeqq(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x29, true, -1); }
void vpcmpgtb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x64, true, -1); }
void vpcmpgtb(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x64, true, -1); }
void vpcmpgtw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x65, true, -1); }
void vpcmpgtw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x65, true, -1); }
void vpcmpgtd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x66, true, -1); }
void vpcmpgtd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x66, true, -1); }
void vpcmpgtq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x37, true, -1); }
void vpcmpgtq(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x37, true, -1); }
void vphaddw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x01, true, -1); }
void vphaddw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x01, true, -1); }
void vphaddd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x02, true, -1); }
void vphaddd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x02, true, -1); }
void vphaddsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x03, true, -1); }
void vphaddsw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x03, true, -1); }
void vphsubw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x05, true, -1); }
void vphsubw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x05, true, -1); }
void vphsubd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x06, true, -1); }
void vphsubd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x06, true, -1); }
void vphsubsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x07, true, -1); }
void vphsubsw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x07, true, -1); }
void vpmaddwd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xF5, true, -1); }
void vpmaddwd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xF5, true, -1); }
void vpmaddubsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x04, true, -1); }
void vpmaddubsw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x04, true, -1); }
void vpmaxsb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x3C, true, -1); }
void vpmaxsb(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x3C, true, -1); }
void vpmaxsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xEE, true, -1); }
void vpmaxsw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xEE, true, -1); }
void vpmaxsd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x3D, true, -1); }
void vpmaxsd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x3D, true, -1); }
void vpmaxub(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xDE, true, -1); }
void vpmaxub(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xDE, true, -1); }
void vpmaxuw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x3E, true, -1); }
void vpmaxuw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x3E, true, -1); }
void vpmaxud(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x3F, true, -1); }
void vpmaxud(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x3F, true, -1); }
void vpminsb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x38, true, -1); }
void vpminsb(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x38, true, -1); }
void vpminsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xEA, true, -1); }
void vpminsw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xEA, true, -1); }
void vpminsd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x39, true, -1); }
void vpminsd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x39, true, -1); }
void vpminub(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xDA, true, -1); }
void vpminub(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xDA, true, -1); }
void vpminuw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x3A, true, -1); }
void vpminuw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x3A, true, -1); }
void vpminud(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x3B, true, -1); }
void vpminud(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x3B, true, -1); }
void vpmulhuw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xE4, true, -1); }
void vpmulhuw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xE4, true, -1); }
void vpmulhrsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x0B, true, -1); }
void vpmulhrsw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x0B, true, -1); }
void vpmulhw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xE5, true, -1); }
void vpmulhw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xE5, true, -1); }
void vpmullw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xD5, true, -1); }
void vpmullw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xD5, true, -1); }
void vpmulld(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x40, true, -1); }
void vpmulld(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x40, true, -1); }
void vpmuludq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xF4, false, -1); }
void vpmuludq(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xF4, false, -1); }
void vpmuldq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x28, true, -1); }
void vpmuldq(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x28, true, -1); }
void vpor(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xEB, true, -1); }
void vpor(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xEB, true, -1); }
void vpsadbw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xF6, true, -1); }
void vpsadbw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xF6, true, -1); }
void vpshufb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x00, true, -1); }
void vpsignb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x08, true, -1); }
void vpsignb(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x08, true, -1); }
void vpsignw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x09, true, -1); }
void vpsignw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x09, true, -1); }
void vpsignd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F38 | PP_66, 0x0A, true, -1); }
void vpsignd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F38 | PP_66, 0x0A, true, -1); }
void vpsllw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xF1, true, -1); }
void vpsllw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xF1, true, -1); }
void vpslld(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xF2, true, -1); }
void vpslld(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xF2, true, -1); }
void vpsllq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xF3, true, -1); }
void vpsllq(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xF3, true, -1); }
void vpsraw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xE1, true, -1); }
void vpsraw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xE1, true, -1); }
void vpsrad(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xE2, true, -1); }
void vpsrad(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xE2, true, -1); }
void vpsrlw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xD1, true, -1); }
void vpsrlw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xD1, true, -1); }
void vpsrld(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xD2, true, -1); }
void vpsrld(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xD2, true, -1); }
void vpsrlq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xD3, true, -1); }
void vpsrlq(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xD3, true, -1); }
void vpsubb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xF8, true, -1); }
void vpsubb(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xF8, true, -1); }
void vpsubw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xF9, true, -1); }
void vpsubw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xF9, true, -1); }
void vpsubd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xFA, true, -1); }
void vpsubd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xFA, true, -1); }
void vpsubq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xFB, true, -1); }
void vpsubq(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xFB, true, -1); }
void vpsubsb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xE8, true, -1); }
void vpsubsb(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xE8, true, -1); }
void vpsubsw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xE9, true, -1); }
void vpsubsw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xE9, true, -1); }
void vpsubusb(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xD8, true, -1); }
void vpsubusb(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xD8, true, -1); }
void vpsubusw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xD9, true, -1); }
void vpsubusw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xD9, true, -1); }
void vpunpckhbw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x68, true, -1); }
void vpunpckhbw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x68, true, -1); }
void vpunpckhwd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x69, true, -1); }
void vpunpckhwd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x69, true, -1); }
void vpunpckhdq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x6A, true, -1); }
void vpunpckhdq(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x6A, true, -1); }
void vpunpckhqdq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x6D, true, -1); }
void vpunpckhqdq(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x6D, true, -1); }
void vpunpcklbw(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x60, true, -1); }
void vpunpcklbw(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x60, true, -1); }
void vpunpcklwd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x61, true, -1); }
void vpunpcklwd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x61, true, -1); }
void vpunpckldq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x62, true, -1); }
void vpunpckldq(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x62, true, -1); }
void vpunpcklqdq(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x6C, true, -1); }
void vpunpcklqdq(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x6C, true, -1); }
void vpxor(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xEF, true, -1); }
void vpxor(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xEF, true, -1); }
void vrcpss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_F3, 0x53, false, -1); }
void vrcpss(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_F3, 0x53, false, -1); }
void vrsqrtss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_F3, 0x52, false, -1); }
void vrsqrtss(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_F3, 0x52, false, -1); }
void vshufpd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0xC6, true, -1, imm); }
void vshufpd(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0xC6, true, -1, imm); }
void vshufps(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { opAVX_X_X_XM(x1, x2, op, MM_0F, 0xC6, true, -1, imm); }
void vshufps(const Xmm& xmm, const Operand& op, uint8 imm) { opAVX_X_X_XM(xmm, xmm, op, MM_0F, 0xC6, true, -1, imm); }
void vsqrtsd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_F2, 0x51, false, -1); }
void vsqrtsd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_F2, 0x51, false, -1); }
void vsqrtss(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_F3, 0x51, false, -1); }
void vsqrtss(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_F3, 0x51, false, -1); }
void vunpckhpd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x15, true, -1); }
void vunpckhpd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x15, true, -1); }
void vunpckhps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F, 0x15, true, -1); }
void vunpckhps(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F, 0x15, true, -1); }
void vunpcklpd(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F | PP_66, 0x14, true, -1); }
void vunpcklpd(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F | PP_66, 0x14, true, -1); }
void vunpcklps(const Xmm& x1, const Xmm& x2, const Operand& op) { opAVX_X_X_XM(x1, x2, op, MM_0F, 0x14, true, -1); }
void vunpcklps(const Xmm& xmm, const Operand& op) { opAVX_X_X_XM(xmm, xmm, op, MM_0F, 0x14, true, -1); }
void vaeskeygenassist(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, MM_0F3A | PP_66, 0xDF, false, 0, imm); }
void vroundpd(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, MM_0F3A | PP_66, 0x09, true, 0, imm); }
void vroundps(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, MM_0F3A | PP_66, 0x08, true, 0, imm); }
void vpermilpd(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, MM_0F3A | PP_66, 0x05, true, 0, imm); }
void vpermilps(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, MM_0F3A | PP_66, 0x04, true, 0, imm); }
void vpcmpestri(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, MM_0F3A | PP_66, 0x61, false, 0, imm); }
void vpcmpestrm(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, MM_0F3A | PP_66, 0x60, false, 0, imm); }
void vpcmpistri(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, MM_0F3A | PP_66, 0x63, false, 0, imm); }
void vpcmpistrm(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, MM_0F3A | PP_66, 0x62, false, 0, imm); }
void vtestps(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x0E, true, 0); }
void vtestpd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x0F, true, 0); }
void vcomisd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_66, 0x2F, false, -1); }
void vcomiss(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F, 0x2F, false, -1); }
void vcvtdq2ps(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F, 0x5B, true, -1); }
void vcvtps2dq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_66, 0x5B, true, -1); }
void vcvttps2dq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_F3, 0x5B, true, -1); }
void vmovapd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_66, 0x28, true, -1); }
void vmovaps(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F, 0x28, true, -1); }
void vmovddup(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_F2, 0x12, true, -1); }
void vmovdqa(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_66, 0x6F, true, -1); }
void vmovdqu(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_F3, 0x6F, true, -1); }
void vmovshdup(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_F3, 0x16, true, -1); }
void vmovsldup(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_F3, 0x12, true, -1); }
void vmovupd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_66, 0x10, true, -1); }
void vmovups(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F, 0x10, true, -1); }
void vpabsb(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x1C, true, -1); }
void vpabsw(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x1D, true, -1); }
void vpabsd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x1E, true, -1); }
void vphminposuw(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x41, false, -1); }
void vpmovsxbw(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x20, true, -1); }
void vpmovsxbd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x21, true, -1); }
void vpmovsxbq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x22, true, -1); }
void vpmovsxwd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x23, true, -1); }
void vpmovsxwq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x24, true, -1); }
void vpmovsxdq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x25, true, -1); }
void vpmovzxbw(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x30, true, -1); }
void vpmovzxbd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x31, true, -1); }
void vpmovzxbq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x32, true, -1); }
void vpmovzxwd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x33, true, -1); }
void vpmovzxwq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x34, true, -1); }
void vpmovzxdq(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x35, true, -1); }
void vpshufd(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_66, 0x70, true, -1, imm); }
void vpshufhw(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_F3, 0x70, true, -1, imm); }
void vpshuflw(const Xmm& xm, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_F2, 0x70, true, -1, imm); }
void vptest(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F38 | PP_66, 0x17, false, -1); }
void vrcpps(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F, 0x53, true, -1); }
void vrsqrtps(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F, 0x52, true, -1); }
void vsqrtpd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_66, 0x51, true, -1); }
void vsqrtps(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F, 0x51, true, -1); }
void vucomisd(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F | PP_66, 0x2E, false, -1); }
void vucomiss(const Xmm& xm, const Operand& op) { opAVX_X_XM_IMM(xm, op, MM_0F, 0x2E, false, -1); }
void vmovapd(const Address& addr, const Xmm& xmm) { opAVX_X_XM_IMM(xmm, addr, MM_0F | PP_66, 0x29, true, -1); }
void vmovaps(const Address& addr, const Xmm& xmm) { opAVX_X_XM_IMM(xmm, addr, MM_0F, 0x29, true, -1); }
void vmovdqa(const Address& addr, const Xmm& xmm) { opAVX_X_XM_IMM(xmm, addr, MM_0F | PP_66, 0x7F, true, -1); }
void vmovdqu(const Address& addr, const Xmm& xmm) { opAVX_X_XM_IMM(xmm, addr, MM_0F | PP_F3, 0x7F, true, -1); }
void vmovupd(const Address& addr, const Xmm& xmm) { opAVX_X_XM_IMM(xmm, addr, MM_0F | PP_66, 0x11, true, -1); }
void vmovups(const Address& addr, const Xmm& xmm) { opAVX_X_XM_IMM(xmm, addr, MM_0F, 0x11, true, -1); }
void vaddsubpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_66, 0xD0, true, -1); }
void vaddsubps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F2, 0xD0, true, -1); }
void vhaddpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_66, 0x7C, true, -1); }
void vhaddps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F2, 0x7C, true, -1); }
void vhsubpd(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_66, 0x7D, true, -1); }
void vhsubps(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F | PP_F2, 0x7D, true, -1); }
void vaesenc(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xDC, false, 0); }
void vaesenclast(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xDD, false, 0); }
void vaesdec(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xDE, false, 0); }
void vaesdeclast(const Xmm& xmm, const Operand& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xDF, false, 0); }
void vmaskmovps(const Xmm& x1, const Xmm& x2, const Address& addr) { opAVX_X_X_XM(x1, x2, addr, MM_0F38 | PP_66, 0x2C, true, 0); }
void vmaskmovps(const Address& addr, const Xmm& x1, const Xmm& x2) { opAVX_X_X_XM(x2, x1, addr, MM_0F38 | PP_66, 0x2E, true, 0); }
void vmaskmovpd(const Xmm& x1, const Xmm& x2, const Address& addr) { opAVX_X_X_XM(x1, x2, addr, MM_0F38 | PP_66, 0x2D, true, 0); }
void vmaskmovpd(const Address& addr, const Xmm& x1, const Xmm& x2) { opAVX_X_X_XM(x2, x1, addr, MM_0F38 | PP_66, 0x2F, true, 0); }
void vpmaskmovd(const Xmm& x1, const Xmm& x2, const Address& addr) { opAVX_X_X_XM(x1, x2, addr, MM_0F38 | PP_66, 0x8C, true, 0); }
void vpmaskmovd(const Address& addr, const Xmm& x1, const Xmm& x2) { opAVX_X_X_XM(x2, x1, addr, MM_0F38 | PP_66, 0x8E, true, 0); }
void vpmaskmovq(const Xmm& x1, const Xmm& x2, const Address& addr) { opAVX_X_X_XM(x1, x2, addr, MM_0F38 | PP_66, 0x8C, true, 1); }
void vpmaskmovq(const Address& addr, const Xmm& x1, const Xmm& x2) { opAVX_X_X_XM(x2, x1, addr, MM_0F38 | PP_66, 0x8E, true, 1); }
void vpermd(const Ymm& y1, const Ymm& y2, const Operand& op) { opAVX_X_X_XM(y1, y2, op, MM_0F38 | PP_66, 0x36, true, 0); }
void vpermps(const Ymm& y1, const Ymm& y2, const Operand& op) { opAVX_X_X_XM(y1, y2, op, MM_0F38 | PP_66, 0x16, true, 0); }
void vpermq(const Ymm& y, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(y, op, MM_0F3A | PP_66, 0x00, true, 1, imm); }
void vpermpd(const Ymm& y, const Operand& op, uint8 imm) { opAVX_X_XM_IMM(y, op, MM_0F3A | PP_66, 0x01, true, 1, imm); }
void cmpeqpd(const Xmm& x, const Operand& op) { cmppd(x, op, 0); }
void vcmpeqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 0); }
void vcmpeqpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 0); }
void cmpltpd(const Xmm& x, const Operand& op) { cmppd(x, op, 1); }
void vcmpltpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 1); }
void vcmpltpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 1); }
void cmplepd(const Xmm& x, const Operand& op) { cmppd(x, op, 2); }
void vcmplepd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 2); }
void vcmplepd(const Xmm& x, const Operand& op) { vcmppd(x, op, 2); }
void cmpunordpd(const Xmm& x, const Operand& op) { cmppd(x, op, 3); }
void vcmpunordpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 3); }
void vcmpunordpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 3); }
void cmpneqpd(const Xmm& x, const Operand& op) { cmppd(x, op, 4); }
void vcmpneqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 4); }
void vcmpneqpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 4); }
void cmpnltpd(const Xmm& x, const Operand& op) { cmppd(x, op, 5); }
void vcmpnltpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 5); }
void vcmpnltpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 5); }
void cmpnlepd(const Xmm& x, const Operand& op) { cmppd(x, op, 6); }
void vcmpnlepd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 6); }
void vcmpnlepd(const Xmm& x, const Operand& op) { vcmppd(x, op, 6); }
void cmpordpd(const Xmm& x, const Operand& op) { cmppd(x, op, 7); }
void vcmpordpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 7); }
void vcmpordpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 7); }
void vcmpeq_uqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 8); }
void vcmpeq_uqpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 8); }
void vcmpngepd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 9); }
void vcmpngepd(const Xmm& x, const Operand& op) { vcmppd(x, op, 9); }
void vcmpngtpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 10); }
void vcmpngtpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 10); }
void vcmpfalsepd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 11); }
void vcmpfalsepd(const Xmm& x, const Operand& op) { vcmppd(x, op, 11); }
void vcmpneq_oqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 12); }
void vcmpneq_oqpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 12); }
void vcmpgepd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 13); }
void vcmpgepd(const Xmm& x, const Operand& op) { vcmppd(x, op, 13); }
void vcmpgtpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 14); }
void vcmpgtpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 14); }
void vcmptruepd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 15); }
void vcmptruepd(const Xmm& x, const Operand& op) { vcmppd(x, op, 15); }
void vcmpeq_ospd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 16); }
void vcmpeq_ospd(const Xmm& x, const Operand& op) { vcmppd(x, op, 16); }
void vcmplt_oqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 17); }
void vcmplt_oqpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 17); }
void vcmple_oqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 18); }
void vcmple_oqpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 18); }
void vcmpunord_spd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 19); }
void vcmpunord_spd(const Xmm& x, const Operand& op) { vcmppd(x, op, 19); }
void vcmpneq_uspd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 20); }
void vcmpneq_uspd(const Xmm& x, const Operand& op) { vcmppd(x, op, 20); }
void vcmpnlt_uqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 21); }
void vcmpnlt_uqpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 21); }
void vcmpnle_uqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 22); }
void vcmpnle_uqpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 22); }
void vcmpord_spd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 23); }
void vcmpord_spd(const Xmm& x, const Operand& op) { vcmppd(x, op, 23); }
void vcmpeq_uspd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 24); }
void vcmpeq_uspd(const Xmm& x, const Operand& op) { vcmppd(x, op, 24); }
void vcmpnge_uqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 25); }
void vcmpnge_uqpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 25); }
void vcmpngt_uqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 26); }
void vcmpngt_uqpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 26); }
void vcmpfalse_ospd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 27); }
void vcmpfalse_ospd(const Xmm& x, const Operand& op) { vcmppd(x, op, 27); }
void vcmpneq_ospd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 28); }
void vcmpneq_ospd(const Xmm& x, const Operand& op) { vcmppd(x, op, 28); }
void vcmpge_oqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 29); }
void vcmpge_oqpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 29); }
void vcmpgt_oqpd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 30); }
void vcmpgt_oqpd(const Xmm& x, const Operand& op) { vcmppd(x, op, 30); }
void vcmptrue_uspd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmppd(x1, x2, op, 31); }
void vcmptrue_uspd(const Xmm& x, const Operand& op) { vcmppd(x, op, 31); }
void cmpeqps(const Xmm& x, const Operand& op) { cmpps(x, op, 0); }
void vcmpeqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 0); }
void vcmpeqps(const Xmm& x, const Operand& op) { vcmpps(x, op, 0); }
void cmpltps(const Xmm& x, const Operand& op) { cmpps(x, op, 1); }
void vcmpltps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 1); }
void vcmpltps(const Xmm& x, const Operand& op) { vcmpps(x, op, 1); }
void cmpleps(const Xmm& x, const Operand& op) { cmpps(x, op, 2); }
void vcmpleps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 2); }
void vcmpleps(const Xmm& x, const Operand& op) { vcmpps(x, op, 2); }
void cmpunordps(const Xmm& x, const Operand& op) { cmpps(x, op, 3); }
void vcmpunordps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 3); }
void vcmpunordps(const Xmm& x, const Operand& op) { vcmpps(x, op, 3); }
void cmpneqps(const Xmm& x, const Operand& op) { cmpps(x, op, 4); }
void vcmpneqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 4); }
void vcmpneqps(const Xmm& x, const Operand& op) { vcmpps(x, op, 4); }
void cmpnltps(const Xmm& x, const Operand& op) { cmpps(x, op, 5); }
void vcmpnltps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 5); }
void vcmpnltps(const Xmm& x, const Operand& op) { vcmpps(x, op, 5); }
void cmpnleps(const Xmm& x, const Operand& op) { cmpps(x, op, 6); }
void vcmpnleps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 6); }
void vcmpnleps(const Xmm& x, const Operand& op) { vcmpps(x, op, 6); }
void cmpordps(const Xmm& x, const Operand& op) { cmpps(x, op, 7); }
void vcmpordps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 7); }
void vcmpordps(const Xmm& x, const Operand& op) { vcmpps(x, op, 7); }
void vcmpeq_uqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 8); }
void vcmpeq_uqps(const Xmm& x, const Operand& op) { vcmpps(x, op, 8); }
void vcmpngeps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 9); }
void vcmpngeps(const Xmm& x, const Operand& op) { vcmpps(x, op, 9); }
void vcmpngtps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 10); }
void vcmpngtps(const Xmm& x, const Operand& op) { vcmpps(x, op, 10); }
void vcmpfalseps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 11); }
void vcmpfalseps(const Xmm& x, const Operand& op) { vcmpps(x, op, 11); }
void vcmpneq_oqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 12); }
void vcmpneq_oqps(const Xmm& x, const Operand& op) { vcmpps(x, op, 12); }
void vcmpgeps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 13); }
void vcmpgeps(const Xmm& x, const Operand& op) { vcmpps(x, op, 13); }
void vcmpgtps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 14); }
void vcmpgtps(const Xmm& x, const Operand& op) { vcmpps(x, op, 14); }
void vcmptrueps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 15); }
void vcmptrueps(const Xmm& x, const Operand& op) { vcmpps(x, op, 15); }
void vcmpeq_osps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 16); }
void vcmpeq_osps(const Xmm& x, const Operand& op) { vcmpps(x, op, 16); }
void vcmplt_oqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 17); }
void vcmplt_oqps(const Xmm& x, const Operand& op) { vcmpps(x, op, 17); }
void vcmple_oqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 18); }
void vcmple_oqps(const Xmm& x, const Operand& op) { vcmpps(x, op, 18); }
void vcmpunord_sps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 19); }
void vcmpunord_sps(const Xmm& x, const Operand& op) { vcmpps(x, op, 19); }
void vcmpneq_usps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 20); }
void vcmpneq_usps(const Xmm& x, const Operand& op) { vcmpps(x, op, 20); }
void vcmpnlt_uqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 21); }
void vcmpnlt_uqps(const Xmm& x, const Operand& op) { vcmpps(x, op, 21); }
void vcmpnle_uqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 22); }
void vcmpnle_uqps(const Xmm& x, const Operand& op) { vcmpps(x, op, 22); }
void vcmpord_sps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 23); }
void vcmpord_sps(const Xmm& x, const Operand& op) { vcmpps(x, op, 23); }
void vcmpeq_usps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 24); }
void vcmpeq_usps(const Xmm& x, const Operand& op) { vcmpps(x, op, 24); }
void vcmpnge_uqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 25); }
void vcmpnge_uqps(const Xmm& x, const Operand& op) { vcmpps(x, op, 25); }
void vcmpngt_uqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 26); }
void vcmpngt_uqps(const Xmm& x, const Operand& op) { vcmpps(x, op, 26); }
void vcmpfalse_osps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 27); }
void vcmpfalse_osps(const Xmm& x, const Operand& op) { vcmpps(x, op, 27); }
void vcmpneq_osps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 28); }
void vcmpneq_osps(const Xmm& x, const Operand& op) { vcmpps(x, op, 28); }
void vcmpge_oqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 29); }
void vcmpge_oqps(const Xmm& x, const Operand& op) { vcmpps(x, op, 29); }
void vcmpgt_oqps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 30); }
void vcmpgt_oqps(const Xmm& x, const Operand& op) { vcmpps(x, op, 30); }
void vcmptrue_usps(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpps(x1, x2, op, 31); }
void vcmptrue_usps(const Xmm& x, const Operand& op) { vcmpps(x, op, 31); }
void cmpeqsd(const Xmm& x, const Operand& op) { cmpsd(x, op, 0); }
void vcmpeqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 0); }
void vcmpeqsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 0); }
void cmpltsd(const Xmm& x, const Operand& op) { cmpsd(x, op, 1); }
void vcmpltsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 1); }
void vcmpltsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 1); }
void cmplesd(const Xmm& x, const Operand& op) { cmpsd(x, op, 2); }
void vcmplesd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 2); }
void vcmplesd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 2); }
void cmpunordsd(const Xmm& x, const Operand& op) { cmpsd(x, op, 3); }
void vcmpunordsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 3); }
void vcmpunordsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 3); }
void cmpneqsd(const Xmm& x, const Operand& op) { cmpsd(x, op, 4); }
void vcmpneqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 4); }
void vcmpneqsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 4); }
void cmpnltsd(const Xmm& x, const Operand& op) { cmpsd(x, op, 5); }
void vcmpnltsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 5); }
void vcmpnltsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 5); }
void cmpnlesd(const Xmm& x, const Operand& op) { cmpsd(x, op, 6); }
void vcmpnlesd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 6); }
void vcmpnlesd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 6); }
void cmpordsd(const Xmm& x, const Operand& op) { cmpsd(x, op, 7); }
void vcmpordsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 7); }
void vcmpordsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 7); }
void vcmpeq_uqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 8); }
void vcmpeq_uqsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 8); }
void vcmpngesd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 9); }
void vcmpngesd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 9); }
void vcmpngtsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 10); }
void vcmpngtsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 10); }
void vcmpfalsesd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 11); }
void vcmpfalsesd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 11); }
void vcmpneq_oqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 12); }
void vcmpneq_oqsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 12); }
void vcmpgesd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 13); }
void vcmpgesd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 13); }
void vcmpgtsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 14); }
void vcmpgtsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 14); }
void vcmptruesd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 15); }
void vcmptruesd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 15); }
void vcmpeq_ossd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 16); }
void vcmpeq_ossd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 16); }
void vcmplt_oqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 17); }
void vcmplt_oqsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 17); }
void vcmple_oqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 18); }
void vcmple_oqsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 18); }
void vcmpunord_ssd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 19); }
void vcmpunord_ssd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 19); }
void vcmpneq_ussd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 20); }
void vcmpneq_ussd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 20); }
void vcmpnlt_uqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 21); }
void vcmpnlt_uqsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 21); }
void vcmpnle_uqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 22); }
void vcmpnle_uqsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 22); }
void vcmpord_ssd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 23); }
void vcmpord_ssd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 23); }
void vcmpeq_ussd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 24); }
void vcmpeq_ussd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 24); }
void vcmpnge_uqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 25); }
void vcmpnge_uqsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 25); }
void vcmpngt_uqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 26); }
void vcmpngt_uqsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 26); }
void vcmpfalse_ossd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 27); }
void vcmpfalse_ossd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 27); }
void vcmpneq_ossd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 28); }
void vcmpneq_ossd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 28); }
void vcmpge_oqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 29); }
void vcmpge_oqsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 29); }
void vcmpgt_oqsd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 30); }
void vcmpgt_oqsd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 30); }
void vcmptrue_ussd(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpsd(x1, x2, op, 31); }
void vcmptrue_ussd(const Xmm& x, const Operand& op) { vcmpsd(x, op, 31); }
void cmpeqss(const Xmm& x, const Operand& op) { cmpss(x, op, 0); }
void vcmpeqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 0); }
void vcmpeqss(const Xmm& x, const Operand& op) { vcmpss(x, op, 0); }
void cmpltss(const Xmm& x, const Operand& op) { cmpss(x, op, 1); }
void vcmpltss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 1); }
void vcmpltss(const Xmm& x, const Operand& op) { vcmpss(x, op, 1); }
void cmpless(const Xmm& x, const Operand& op) { cmpss(x, op, 2); }
void vcmpless(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 2); }
void vcmpless(const Xmm& x, const Operand& op) { vcmpss(x, op, 2); }
void cmpunordss(const Xmm& x, const Operand& op) { cmpss(x, op, 3); }
void vcmpunordss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 3); }
void vcmpunordss(const Xmm& x, const Operand& op) { vcmpss(x, op, 3); }
void cmpneqss(const Xmm& x, const Operand& op) { cmpss(x, op, 4); }
void vcmpneqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 4); }
void vcmpneqss(const Xmm& x, const Operand& op) { vcmpss(x, op, 4); }
void cmpnltss(const Xmm& x, const Operand& op) { cmpss(x, op, 5); }
void vcmpnltss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 5); }
void vcmpnltss(const Xmm& x, const Operand& op) { vcmpss(x, op, 5); }
void cmpnless(const Xmm& x, const Operand& op) { cmpss(x, op, 6); }
void vcmpnless(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 6); }
void vcmpnless(const Xmm& x, const Operand& op) { vcmpss(x, op, 6); }
void cmpordss(const Xmm& x, const Operand& op) { cmpss(x, op, 7); }
void vcmpordss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 7); }
void vcmpordss(const Xmm& x, const Operand& op) { vcmpss(x, op, 7); }
void vcmpeq_uqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 8); }
void vcmpeq_uqss(const Xmm& x, const Operand& op) { vcmpss(x, op, 8); }
void vcmpngess(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 9); }
void vcmpngess(const Xmm& x, const Operand& op) { vcmpss(x, op, 9); }
void vcmpngtss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 10); }
void vcmpngtss(const Xmm& x, const Operand& op) { vcmpss(x, op, 10); }
void vcmpfalsess(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 11); }
void vcmpfalsess(const Xmm& x, const Operand& op) { vcmpss(x, op, 11); }
void vcmpneq_oqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 12); }
void vcmpneq_oqss(const Xmm& x, const Operand& op) { vcmpss(x, op, 12); }
void vcmpgess(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 13); }
void vcmpgess(const Xmm& x, const Operand& op) { vcmpss(x, op, 13); }
void vcmpgtss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 14); }
void vcmpgtss(const Xmm& x, const Operand& op) { vcmpss(x, op, 14); }
void vcmptruess(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 15); }
void vcmptruess(const Xmm& x, const Operand& op) { vcmpss(x, op, 15); }
void vcmpeq_osss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 16); }
void vcmpeq_osss(const Xmm& x, const Operand& op) { vcmpss(x, op, 16); }
void vcmplt_oqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 17); }
void vcmplt_oqss(const Xmm& x, const Operand& op) { vcmpss(x, op, 17); }
void vcmple_oqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 18); }
void vcmple_oqss(const Xmm& x, const Operand& op) { vcmpss(x, op, 18); }
void vcmpunord_sss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 19); }
void vcmpunord_sss(const Xmm& x, const Operand& op) { vcmpss(x, op, 19); }
void vcmpneq_usss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 20); }
void vcmpneq_usss(const Xmm& x, const Operand& op) { vcmpss(x, op, 20); }
void vcmpnlt_uqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 21); }
void vcmpnlt_uqss(const Xmm& x, const Operand& op) { vcmpss(x, op, 21); }
void vcmpnle_uqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 22); }
void vcmpnle_uqss(const Xmm& x, const Operand& op) { vcmpss(x, op, 22); }
void vcmpord_sss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 23); }
void vcmpord_sss(const Xmm& x, const Operand& op) { vcmpss(x, op, 23); }
void vcmpeq_usss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 24); }
void vcmpeq_usss(const Xmm& x, const Operand& op) { vcmpss(x, op, 24); }
void vcmpnge_uqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 25); }
void vcmpnge_uqss(const Xmm& x, const Operand& op) { vcmpss(x, op, 25); }
void vcmpngt_uqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 26); }
void vcmpngt_uqss(const Xmm& x, const Operand& op) { vcmpss(x, op, 26); }
void vcmpfalse_osss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 27); }
void vcmpfalse_osss(const Xmm& x, const Operand& op) { vcmpss(x, op, 27); }
void vcmpneq_osss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 28); }
void vcmpneq_osss(const Xmm& x, const Operand& op) { vcmpss(x, op, 28); }
void vcmpge_oqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 29); }
void vcmpge_oqss(const Xmm& x, const Operand& op) { vcmpss(x, op, 29); }
void vcmpgt_oqss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 30); }
void vcmpgt_oqss(const Xmm& x, const Operand& op) { vcmpss(x, op, 30); }
void vcmptrue_usss(const Xmm& x1, const Xmm& x2, const Operand& op) { vcmpss(x1, x2, op, 31); }
void vcmptrue_usss(const Xmm& x, const Operand& op) { vcmpss(x, op, 31); }
void vmovhpd(const Xmm& x, const Operand& op1, const Operand& op2 = Operand()) { if (!op2.isNone() && !op2.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x, op1, op2, MM_0F | PP_66, 0x16, false); }
void vmovhpd(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, MM_0F | PP_66, 0x17, false); }
void vmovhps(const Xmm& x, const Operand& op1, const Operand& op2 = Operand()) { if (!op2.isNone() && !op2.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x, op1, op2, MM_0F, 0x16, false); }
void vmovhps(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, MM_0F, 0x17, false); }
void vmovlpd(const Xmm& x, const Operand& op1, const Operand& op2 = Operand()) { if (!op2.isNone() && !op2.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x, op1, op2, MM_0F | PP_66, 0x12, false); }
void vmovlpd(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, MM_0F | PP_66, 0x13, false); }
void vmovlps(const Xmm& x, const Operand& op1, const Operand& op2 = Operand()) { if (!op2.isNone() && !op2.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x, op1, op2, MM_0F, 0x12, false); }
void vmovlps(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, MM_0F, 0x13, false); }
void vfmadd132pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x98, true, 1); }
void vfmadd213pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xA8, true, 1); }
void vfmadd231pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xB8, true, 1); }
void vfmadd132ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x98, true, 0); }
void vfmadd213ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xA8, true, 0); }
void vfmadd231ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xB8, true, 0); }
void vfmadd132sd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x99, false, 1); }
void vfmadd213sd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xA9, false, 1); }
void vfmadd231sd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xB9, false, 1); }
void vfmadd132ss(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x99, false, 0); }
void vfmadd213ss(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xA9, false, 0); }
void vfmadd231ss(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xB9, false, 0); }
void vfmaddsub132pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x96, true, 1); }
void vfmaddsub213pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xA6, true, 1); }
void vfmaddsub231pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xB6, true, 1); }
void vfmaddsub132ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x96, true, 0); }
void vfmaddsub213ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xA6, true, 0); }
void vfmaddsub231ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xB6, true, 0); }
void vfmsubadd132pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x97, true, 1); }
void vfmsubadd213pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xA7, true, 1); }
void vfmsubadd231pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xB7, true, 1); }
void vfmsubadd132ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x97, true, 0); }
void vfmsubadd213ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xA7, true, 0); }
void vfmsubadd231ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xB7, true, 0); }
void vfmsub132pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x9A, true, 1); }
void vfmsub213pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xAA, true, 1); }
void vfmsub231pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xBA, true, 1); }
void vfmsub132ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x9A, true, 0); }
void vfmsub213ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xAA, true, 0); }
void vfmsub231ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xBA, true, 0); }
void vfmsub132sd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x9B, false, 1); }
void vfmsub213sd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xAB, false, 1); }
void vfmsub231sd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xBB, false, 1); }
void vfmsub132ss(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x9B, false, 0); }
void vfmsub213ss(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xAB, false, 0); }
void vfmsub231ss(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xBB, false, 0); }
void vfnmadd132pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x9C, true, 1); }
void vfnmadd213pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xAC, true, 1); }
void vfnmadd231pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xBC, true, 1); }
void vfnmadd132ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x9C, true, 0); }
void vfnmadd213ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xAC, true, 0); }
void vfnmadd231ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xBC, true, 0); }
void vfnmadd132sd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x9D, false, 1); }
void vfnmadd213sd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xAD, false, 1); }
void vfnmadd231sd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xBD, false, 1); }
void vfnmadd132ss(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x9D, false, 0); }
void vfnmadd213ss(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xAD, false, 0); }
void vfnmadd231ss(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xBD, false, 0); }
void vfnmsub132pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x9E, true, 1); }
void vfnmsub213pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xAE, true, 1); }
void vfnmsub231pd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xBE, true, 1); }
void vfnmsub132ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x9E, true, 0); }
void vfnmsub213ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xAE, true, 0); }
void vfnmsub231ps(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xBE, true, 0); }
void vfnmsub132sd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x9F, false, 1); }
void vfnmsub213sd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xAF, false, 1); }
void vfnmsub231sd(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xBF, false, 1); }
void vfnmsub132ss(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0x9F, false, 0); }
void vfnmsub213ss(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xAF, false, 0); }
void vfnmsub231ss(const Xmm& xmm, const Xmm& op1, const Operand& op2 = Operand()) { opAVX_X_X_XM(xmm, op1, op2, MM_0F38 | PP_66, 0xBF, false, 0); }
void vaesimc(const Xmm& x, const Operand& op) { opAVX_X_XM_IMM(x, op, MM_0F38 | PP_66, 0xDB, false, 0); }
void vbroadcastf128(const Ymm& y, const Address& addr) { opAVX_X_XM_IMM(y, addr, MM_0F38 | PP_66, 0x1A, true, 0); }
void vbroadcasti128(const Ymm& y, const Address& addr) { opAVX_X_XM_IMM(y, addr, MM_0F38 | PP_66, 0x5A, true, 0); }
void vbroadcastsd(const Ymm& y, const Operand& op) { if (!(op.isXMM() || op.isMEM())) throw Error(ERR_BAD_COMBINATION); opAVX_X_XM_IMM(y, op, MM_0F38 | PP_66, 0x19, true, 0); }
void vbroadcastss(const Xmm& x, const Operand& op) { if (!(op.isXMM() || op.isMEM())) throw Error(ERR_BAD_COMBINATION); opAVX_X_XM_IMM(x, op, MM_0F38 | PP_66, 0x18, true, 0); }
void vpbroadcastb(const Xmm& x, const Operand& op) { if (!(op.isXMM() || op.isMEM())) throw Error(ERR_BAD_COMBINATION); opAVX_X_XM_IMM(x, op, MM_0F38 | PP_66, 0x78, true, 0); }
void vpbroadcastw(const Xmm& x, const Operand& op) { if (!(op.isXMM() || op.isMEM())) throw Error(ERR_BAD_COMBINATION); opAVX_X_XM_IMM(x, op, MM_0F38 | PP_66, 0x79, true, 0); }
void vpbroadcastd(const Xmm& x, const Operand& op) { if (!(op.isXMM() || op.isMEM())) throw Error(ERR_BAD_COMBINATION); opAVX_X_XM_IMM(x, op, MM_0F38 | PP_66, 0x58, true, 0); }
void vpbroadcastq(const Xmm& x, const Operand& op) { if (!(op.isXMM() || op.isMEM())) throw Error(ERR_BAD_COMBINATION); opAVX_X_XM_IMM(x, op, MM_0F38 | PP_66, 0x59, true, 0); }
void vextractf128(const Operand& op, const Ymm& y, uint8 imm) { opAVX_X_X_XMcvt(y, y.isXMM() ? xm0 : ym0, op, op.isXMM(), Operand::YMM, MM_0F3A | PP_66, 0x19, true, 0, imm); }
void vextracti128(const Operand& op, const Ymm& y, uint8 imm) { opAVX_X_X_XMcvt(y, y.isXMM() ? xm0 : ym0, op, op.isXMM(), Operand::YMM, MM_0F3A | PP_66, 0x39, true, 0, imm); }
void vextractps(const Operand& op, const Xmm& x, uint8 imm) { if (!(op.isREG(32) || op.isMEM()) || x.isYMM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x, x.isXMM() ? xm0 : ym0, op, op.isREG(), Operand::XMM, MM_0F3A | PP_66, 0x17, false, 0, imm); }
void vinsertf128(const Ymm& y1, const Ymm& y2, const Operand& op, uint8 imm) { opAVX_X_X_XMcvt(y1, y2, op, op.isXMM(), Operand::YMM, MM_0F3A | PP_66, 0x18, true, 0, imm); }
void vinserti128(const Ymm& y1, const Ymm& y2, const Operand& op, uint8 imm) { opAVX_X_X_XMcvt(y1, y2, op, op.isXMM(), Operand::YMM, MM_0F3A | PP_66, 0x38, true, 0, imm); }
void vperm2f128(const Ymm& y1, const Ymm& y2, const Operand& op, uint8 imm) { opAVX_X_X_XM(y1, y2, op, MM_0F3A | PP_66, 0x06, true, 0, imm); }
void vperm2i128(const Ymm& y1, const Ymm& y2, const Operand& op, uint8 imm) { opAVX_X_X_XM(y1, y2, op, MM_0F3A | PP_66, 0x46, true, 0, imm); }
void vlddqu(const Xmm& x, const Address& addr) { opAVX_X_X_XM(x, x.isXMM() ? xm0 : ym0, addr, MM_0F | PP_F2, 0xF0, true, 0); }
void vldmxcsr(const Address& addr) { opAVX_X_X_XM(xm2, xm0, addr, MM_0F, 0xAE, false, -1); }
void vstmxcsr(const Address& addr) { opAVX_X_X_XM(xm3, xm0, addr, MM_0F, 0xAE, false, -1); }
void vmaskmovdqu(const Xmm& x1, const Xmm& x2) { opAVX_X_X_XM(x1, xm0, x2, MM_0F | PP_66, 0xF7, false, -1); }
void vpextrb(const Operand& op, const Xmm& x, uint8 imm) { if (!op.isREG(i32e) && !op.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x, xm0, op, !op.isMEM(), Operand::XMM, MM_0F3A | PP_66, 0x14, false, -1, imm); }
void vpextrw(const Reg& r, const Xmm& x, uint8 imm) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, x, MM_0F | PP_66, 0xC5, false, r.isBit(64) ? 1 : 0, imm); }
void vpextrw(const Address& addr, const Xmm& x, uint8 imm) { opAVX_X_X_XM(x, xm0, addr, MM_0F3A | PP_66, 0x15, false, -1, imm); }
void vpextrd(const Operand& op, const Xmm& x, uint8 imm) { if (!op.isREG(32) && !op.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x, xm0, op, !op.isMEM(), Operand::XMM, MM_0F3A | PP_66, 0x16, false, 0, imm); }
void vpinsrb(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { if (!op.isREG(32) && !op.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x1, x2, op, !op.isMEM(), Operand::XMM, MM_0F3A | PP_66, 0x20, false, -1, imm); }
void vpinsrb(const Xmm& x, const Operand& op, uint8 imm) { if (!op.isREG(32) && !op.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x, x, op, !op.isMEM(), Operand::XMM, MM_0F3A | PP_66, 0x20, false, -1, imm); }
void vpinsrw(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { if (!op.isREG(32) && !op.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x1, x2, op, !op.isMEM(), Operand::XMM, MM_0F | PP_66, 0xC4, false, -1, imm); }
void vpinsrw(const Xmm& x, const Operand& op, uint8 imm) { if (!op.isREG(32) && !op.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x, x, op, !op.isMEM(), Operand::XMM, MM_0F | PP_66, 0xC4, false, -1, imm); }
void vpinsrd(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { if (!op.isREG(32) && !op.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x1, x2, op, !op.isMEM(), Operand::XMM, MM_0F3A | PP_66, 0x22, false, 0, imm); }
void vpinsrd(const Xmm& x, const Operand& op, uint8 imm) { if (!op.isREG(32) && !op.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x, x, op, !op.isMEM(), Operand::XMM, MM_0F3A | PP_66, 0x22, false, 0, imm); }
void vpmovmskb(const Reg32e& r, const Xmm& x) { bool isYMM= x.isYMM(); opAVX_X_X_XM(isYMM ? Ymm(r.getIdx()) : Xmm(r.getIdx()), isYMM ? ym0 : xm0, x, MM_0F | PP_66, 0xD7, true); }
void vpslldq(const Xmm& x1, const Xmm& x2, uint8 imm) { opAVX_X_X_XM(x1.isYMM() ? ym7 : xm7, x1, x2, MM_0F | PP_66, 0x73, true, -1, imm); }
void vpslldq(const Xmm& x, uint8 imm) { opAVX_X_X_XM(x.isYMM() ? ym7 : xm7, x, x, MM_0F | PP_66, 0x73, true, -1, imm); }
void vpsrldq(const Xmm& x1, const Xmm& x2, uint8 imm) { opAVX_X_X_XM(x1.isYMM() ? ym3 : xm3, x1, x2, MM_0F | PP_66, 0x73, true, -1, imm); }
void vpsrldq(const Xmm& x, uint8 imm) { opAVX_X_X_XM(x.isYMM() ? ym3 : xm3, x, x, MM_0F | PP_66, 0x73, true, -1, imm); }
void vpsllw(const Xmm& x1, const Xmm& x2, uint8 imm) { opAVX_X_X_XM(x1.isYMM() ? ym6 : xm6, x1, x2, MM_0F | PP_66, 0x71, true, -1, imm); }
void vpsllw(const Xmm& x, uint8 imm) { opAVX_X_X_XM(x.isYMM() ? ym6 : xm6, x, x, MM_0F | PP_66, 0x71, true, -1, imm); }
void vpslld(const Xmm& x1, const Xmm& x2, uint8 imm) { opAVX_X_X_XM(x1.isYMM() ? ym6 : xm6, x1, x2, MM_0F | PP_66, 0x72, true, -1, imm); }
void vpslld(const Xmm& x, uint8 imm) { opAVX_X_X_XM(x.isYMM() ? ym6 : xm6, x, x, MM_0F | PP_66, 0x72, true, -1, imm); }
void vpsllq(const Xmm& x1, const Xmm& x2, uint8 imm) { opAVX_X_X_XM(x1.isYMM() ? ym6 : xm6, x1, x2, MM_0F | PP_66, 0x73, true, -1, imm); }
void vpsllq(const Xmm& x, uint8 imm) { opAVX_X_X_XM(x.isYMM() ? ym6 : xm6, x, x, MM_0F | PP_66, 0x73, true, -1, imm); }
void vpsraw(const Xmm& x1, const Xmm& x2, uint8 imm) { opAVX_X_X_XM(x1.isYMM() ? ym4 : xm4, x1, x2, MM_0F | PP_66, 0x71, true, -1, imm); }
void vpsraw(const Xmm& x, uint8 imm) { opAVX_X_X_XM(x.isYMM() ? ym4 : xm4, x, x, MM_0F | PP_66, 0x71, true, -1, imm); }
void vpsrad(const Xmm& x1, const Xmm& x2, uint8 imm) { opAVX_X_X_XM(x1.isYMM() ? ym4 : xm4, x1, x2, MM_0F | PP_66, 0x72, true, -1, imm); }
void vpsrad(const Xmm& x, uint8 imm) { opAVX_X_X_XM(x.isYMM() ? ym4 : xm4, x, x, MM_0F | PP_66, 0x72, true, -1, imm); }
void vpsrlw(const Xmm& x1, const Xmm& x2, uint8 imm) { opAVX_X_X_XM(x1.isYMM() ? ym2 : xm2, x1, x2, MM_0F | PP_66, 0x71, true, -1, imm); }
void vpsrlw(const Xmm& x, uint8 imm) { opAVX_X_X_XM(x.isYMM() ? ym2 : xm2, x, x, MM_0F | PP_66, 0x71, true, -1, imm); }
void vpsrld(const Xmm& x1, const Xmm& x2, uint8 imm) { opAVX_X_X_XM(x1.isYMM() ? ym2 : xm2, x1, x2, MM_0F | PP_66, 0x72, true, -1, imm); }
void vpsrld(const Xmm& x, uint8 imm) { opAVX_X_X_XM(x.isYMM() ? ym2 : xm2, x, x, MM_0F | PP_66, 0x72, true, -1, imm); }
void vpsrlq(const Xmm& x1, const Xmm& x2, uint8 imm) { opAVX_X_X_XM(x1.isYMM() ? ym2 : xm2, x1, x2, MM_0F | PP_66, 0x73, true, -1, imm); }
void vpsrlq(const Xmm& x, uint8 imm) { opAVX_X_X_XM(x.isYMM() ? ym2 : xm2, x, x, MM_0F | PP_66, 0x73, true, -1, imm); }
void vblendvpd(const Xmm& x1, const Xmm& x2, const Operand& op, const Xmm& x4) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x4B, true, -1, x4.getIdx() << 4); }
void vblendvpd(const Xmm& x1, const Operand& op, const Xmm& x4) { opAVX_X_X_XM(x1, x1, op, MM_0F3A | PP_66, 0x4B, true, -1, x4.getIdx() << 4); }
void vblendvps(const Xmm& x1, const Xmm& x2, const Operand& op, const Xmm& x4) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x4A, true, -1, x4.getIdx() << 4); }
void vblendvps(const Xmm& x1, const Operand& op, const Xmm& x4) { opAVX_X_X_XM(x1, x1, op, MM_0F3A | PP_66, 0x4A, true, -1, x4.getIdx() << 4); }
void vpblendvb(const Xmm& x1, const Xmm& x2, const Operand& op, const Xmm& x4) { opAVX_X_X_XM(x1, x2, op, MM_0F3A | PP_66, 0x4C, false, -1, x4.getIdx() << 4); }
void vpblendvb(const Xmm& x1, const Operand& op, const Xmm& x4) { opAVX_X_X_XM(x1, x1, op, MM_0F3A | PP_66, 0x4C, false, -1, x4.getIdx() << 4); }
void vmovd(const Xmm& x, const Reg32& reg) { opAVX_X_X_XM(x, xm0, Xmm(reg.getIdx()), MM_0F | PP_66, 0x6E, false, 0); }
void vmovd(const Xmm& x, const Address& addr) { opAVX_X_X_XM(x, xm0, addr, MM_0F | PP_66, 0x6E, false, 0); }
void vmovd(const Reg32& reg, const Xmm& x) { opAVX_X_X_XM(x, xm0, Xmm(reg.getIdx()), MM_0F | PP_66, 0x7E, false, 0); }
void vmovd(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, MM_0F | PP_66, 0x7E, false, 0); }
void vmovq(const Xmm& x, const Address& addr) { opAVX_X_X_XM(x, xm0, addr, MM_0F | PP_F3, 0x7E, false, -1); }
void vmovq(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, MM_0F | PP_66, 0xD6, false, -1); }
void vmovq(const Xmm& x1, const Xmm& x2) { opAVX_X_X_XM(x1, xm0, x2, MM_0F | PP_F3, 0x7E, false, -1); }
void vmovhlps(const Xmm& x1, const Xmm& x2, const Operand& op = Operand()) { if (!op.isNone() && !op.isXMM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x1, x2, op, MM_0F, 0x12, false); }
void vmovlhps(const Xmm& x1, const Xmm& x2, const Operand& op = Operand()) { if (!op.isNone() && !op.isXMM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x1, x2, op, MM_0F, 0x16, false); }
void vmovmskpd(const Reg& r, const Xmm& x) { if (!r.isBit(i32e)) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x.isXMM() ? Xmm(r.getIdx()) : Ymm(r.getIdx()), x.isXMM() ? xm0 : ym0, x, MM_0F | PP_66, 0x50, true, 0); }
void vmovmskps(const Reg& r, const Xmm& x) { if (!r.isBit(i32e)) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x.isXMM() ? Xmm(r.getIdx()) : Ymm(r.getIdx()), x.isXMM() ? xm0 : ym0, x, MM_0F, 0x50, true, 0); }
void vmovntdq(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, x.isXMM() ? xm0 : ym0, addr, MM_0F | PP_66, 0xE7, true); }
void vmovntpd(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, x.isXMM() ? xm0 : ym0, addr, MM_0F | PP_66, 0x2B, true); }
void vmovntps(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, x.isXMM() ? xm0 : ym0, addr, MM_0F, 0x2B, true); }
void vmovntdqa(const Xmm& x, const Address& addr) { opAVX_X_X_XM(x, x.isXMM() ? xm0 : ymm0, addr, MM_0F38 | PP_66, 0x2A, true); }
void vmovsd(const Xmm& x1, const Xmm& x2, const Operand& op = Operand()) { if (!op.isNone() && !op.isXMM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x1, x2, op, MM_0F | PP_F2, 0x10, false); }
void vmovsd(const Xmm& x, const Address& addr) { opAVX_X_X_XM(x, xm0, addr, MM_0F | PP_F2, 0x10, false); }
void vmovsd(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, MM_0F | PP_F2, 0x11, false); }
void vmovss(const Xmm& x1, const Xmm& x2, const Operand& op = Operand()) { if (!op.isNone() && !op.isXMM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(x1, x2, op, MM_0F | PP_F3, 0x10, false); }
void vmovss(const Xmm& x, const Address& addr) { opAVX_X_X_XM(x, xm0, addr, MM_0F | PP_F3, 0x10, false); }
void vmovss(const Address& addr, const Xmm& x) { opAVX_X_X_XM(x, xm0, addr, MM_0F | PP_F3, 0x11, false); }
void vcvtss2si(const Reg32& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, MM_0F | PP_F3, 0x2D, false, 0); }
void vcvttss2si(const Reg32& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, MM_0F | PP_F3, 0x2C, false, 0); }
void vcvtsd2si(const Reg32& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, MM_0F | PP_F2, 0x2D, false, 0); }
void vcvttsd2si(const Reg32& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, MM_0F | PP_F2, 0x2C, false, 0); }
void vcvtsi2ss(const Xmm& x, const Operand& op1, const Operand& op2 = Operand()) { if (!op2.isNone() && !(op2.isREG(i32e) || op2.isMEM())) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x, op1, op2, op2.isREG(), Operand::XMM, MM_0F | PP_F3, 0x2A, false, (op1.isMEM() || op2.isMEM()) ? -1 : (op1.isREG(32) || op2.isREG(32)) ? 0 : 1); }
void vcvtsi2sd(const Xmm& x, const Operand& op1, const Operand& op2 = Operand()) { if (!op2.isNone() && !(op2.isREG(i32e) || op2.isMEM())) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x, op1, op2, op2.isREG(), Operand::XMM, MM_0F | PP_F2, 0x2A, false, (op1.isMEM() || op2.isMEM()) ? -1 : (op1.isREG(32) || op2.isREG(32)) ? 0 : 1); }
void vcvtps2pd(const Xmm& x, const Operand& op) { if (!op.isMEM() && !op.isXMM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x, x.isXMM() ? xm0 : ym0, op, !op.isMEM(), x.isXMM() ? Operand::XMM : Operand::YMM, MM_0F, 0x5A, true); }
void vcvtdq2pd(const Xmm& x, const Operand& op) { if (!op.isMEM() && !op.isXMM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x, x.isXMM() ? xm0 : ym0, op, !op.isMEM(), x.isXMM() ? Operand::XMM : Operand::YMM, MM_0F | PP_F3, 0xE6, true); }
void vcvtpd2ps(const Xmm& x, const Operand& op) { if (x.isYMM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(op.isYMM() ? Ymm(x.getIdx()) : x, op.isYMM() ? ym0 : xm0, op, MM_0F | PP_66, 0x5A, true); }
void vcvtpd2dq(const Xmm& x, const Operand& op) { if (x.isYMM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(op.isYMM() ? Ymm(x.getIdx()) : x, op.isYMM() ? ym0 : xm0, op, MM_0F | PP_F2, 0xE6, true); }
void vcvttpd2dq(const Xmm& x, const Operand& op) { if (x.isYMM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XM(op.isYMM() ? Ymm(x.getIdx()) : x, op.isYMM() ? ym0 : xm0, op, MM_0F | PP_66, 0xE6, true); }
void vcvtph2ps(const Xmm& x, const Operand& op) { if (!op.isMEM() && !op.isXMM()) throw Error(ERR_BAD_COMBINATION); opVex(x, NULL, &op, MM_0F38 | PP_66, 0x13, 0); }
void vcvtps2ph(const Operand& op, const Xmm& x, uint8 imm) { if (!op.isMEM() && !op.isXMM()) throw Error(ERR_BAD_COMBINATION); opVex(x, NULL, &op, MM_0F3A | PP_66, 0x1d, 0, imm); }
#ifdef XBYAK64
void vmovq(const Xmm& x, const Reg64& reg) { opAVX_X_X_XM(x, xm0, Xmm(reg.getIdx()), MM_0F | PP_66, 0x6E, false, 1); }
void vmovq(const Reg64& reg, const Xmm& x) { opAVX_X_X_XM(x, xm0, Xmm(reg.getIdx()), MM_0F | PP_66, 0x7E, false, 1); }
void vpextrq(const Operand& op, const Xmm& x, uint8 imm) { if (!op.isREG(64) && !op.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x, xm0, op, !op.isMEM(), Operand::XMM, MM_0F3A | PP_66, 0x16, false, 1, imm); }
void vpinsrq(const Xmm& x1, const Xmm& x2, const Operand& op, uint8 imm) { if (!op.isREG(64) && !op.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x1, x2, op, !op.isMEM(), Operand::XMM, MM_0F3A | PP_66, 0x22, false, 1, imm); }
void vpinsrq(const Xmm& x, const Operand& op, uint8 imm) { if (!op.isREG(64) && !op.isMEM()) throw Error(ERR_BAD_COMBINATION); opAVX_X_X_XMcvt(x, x, op, !op.isMEM(), Operand::XMM, MM_0F3A | PP_66, 0x22, false, 1, imm); }
void vcvtss2si(const Reg64& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, MM_0F | PP_F3, 0x2D, false, 1); }
void vcvttss2si(const Reg64& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, MM_0F | PP_F3, 0x2C, false, 1); }
void vcvtsd2si(const Reg64& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, MM_0F | PP_F2, 0x2D, false, 1); }
void vcvttsd2si(const Reg64& r, const Operand& op) { opAVX_X_X_XM(Xmm(r.getIdx()), xm0, op, MM_0F | PP_F2, 0x2C, false, 1); }
#endif
void andn(const Reg32e& r1, const Reg32e& r2, const Operand& op) { opGpr(r1, r2, op, MM_0F38, 0xf2, true); }
void mulx(const Reg32e& r1, const Reg32e& r2, const Operand& op) { opGpr(r1, r2, op, MM_0F38 | PP_F2, 0xf6, true); }
void pdep(const Reg32e& r1, const Reg32e& r2, const Operand& op) { opGpr(r1, r2, op, MM_0F38 | PP_F2, 0xf5, true); }
void pext(const Reg32e& r1, const Reg32e& r2, const Operand& op) { opGpr(r1, r2, op, MM_0F38 | PP_F3, 0xf5, true); }
void bextr(const Reg32e& r1, const Operand& op, const Reg32e& r2) { opGpr(r1, op, r2, MM_0F38, 0xf7, false); }
void bzhi(const Reg32e& r1, const Operand& op, const Reg32e& r2) { opGpr(r1, op, r2, MM_0F38, 0xf5, false); }
void sarx(const Reg32e& r1, const Operand& op, const Reg32e& r2) { opGpr(r1, op, r2, MM_0F38 | PP_F3, 0xf7, false); }
void shlx(const Reg32e& r1, const Operand& op, const Reg32e& r2) { opGpr(r1, op, r2, MM_0F38 | PP_66, 0xf7, false); }
void shrx(const Reg32e& r1, const Operand& op, const Reg32e& r2) { opGpr(r1, op, r2, MM_0F38 | PP_F2, 0xf7, false); }
void blsi(const Reg32e& r, const Operand& op) { opGpr(Reg32e(3, r.getBit()), op, r, MM_0F38, 0xf3, false); }
void blsmsk(const Reg32e& r, const Operand& op) { opGpr(Reg32e(2, r.getBit()), op, r, MM_0F38, 0xf3, false); }
void blsr(const Reg32e& r, const Operand& op) { opGpr(Reg32e(1, r.getBit()), op, r, MM_0F38, 0xf3, false); }
void vgatherdpd(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, MM_0F38 | PP_66, 0x92, 1, 0); }
void vgatherqpd(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, MM_0F38 | PP_66, 0x93, 1, 1); }
void vgatherdps(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, MM_0F38 | PP_66, 0x92, 0, 1); }
void vgatherqps(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, MM_0F38 | PP_66, 0x93, 0, 2); }
void vpgatherdd(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, MM_0F38 | PP_66, 0x90, 0, 1); }
void vpgatherqd(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, MM_0F38 | PP_66, 0x91, 0, 2); }
void vpgatherdq(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, MM_0F38 | PP_66, 0x90, 1, 0); }
void vpgatherqq(const Xmm& x1, const Address& addr, const Xmm& x2) { opGather(x1, addr, x2, MM_0F38 | PP_66, 0x91, 1, 1); }
