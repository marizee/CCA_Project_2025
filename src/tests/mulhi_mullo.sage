# goal: give some examples of "complicated situations" when splitting
# integer multiplication into mulhi / mullo

WORD_MAX = 2**64 - 1

def split(a, sb):
    # returns (alo, ahi), the low and high parts
    # when splitting `a` at `sb` bits
    alo = a & (2**sb - 1)
    ahi = a >> sb
    return alo, ahi

def split_mul(a, b, sb):
    alo, ahi = split(a, sb)
    blo, bhi = split(b, sb)
    ablo = (alo * blo) & WORD_MAX
    abhi = (ahi * bhi) & WORD_MAX
    abmi = (alo * bhi + ahi * blo) & WORD_MAX
    return ablo, abmi, abhi

def check_split_mul(a, b, sb):
    # warning: we are using python arbitrary-size integers here!
    ab0 = a * b
    ablo, abmi, abhi = split_mul(a, b, sb)
    ab1 = ablo + (abmi << sb) + (abhi << (2*sb))
    return ab0 == ab1

print("Checking a few multiplications:")
for args in [(1,2,10),
             (14121, 51234, 32),
             (ZZ.random_element(2**62), ZZ.random_element(2**62), 32),
             (ZZ.random_element(2**63), ZZ.random_element(2**63), 32),
             (2**63 - 1, 2**63 - 1, 32),
             (2**63 - 1, 2**63 - 1, 30),  # overflow in abhi
             (2**40 - 1, 2**40 - 1, 34),  # overflow in ablo
             (WORD_MAX-15, WORD_MAX-15, 32),  # overflow in abmi
             ]:
    print(*args, check_split_mul(*args))
print("=================")

# ok, but we are in python, so things to care about with C ulong's:
# - we should take sb <= 32 otherwise there could be overflows in multiplications (like alo*blo)
# - if sb == 32, we require that both a and b are < 64 bits (otherwise, abmi may overflow)
# - if sb < 32 we have to be careful about the overflow in ahi*bhi, which brings more constraints
# on the acceptable range for a and b

# say we target 64 bit low/high words of multiplication a*b, for a and b both < 64 bits

def check_split_mullo(a, b, sb):
    ab0 = (a * b) & WORD_MAX
    ablo, abmi, abhi = split_mul(a, b, sb)
    ## ??what is the formula in c file??
    # if sb==32 we expect something like:
    # low word == ablo + (64 low bits of (abmi << sb))
    # --> this may overflow... which is not a problem for mullo! (but is one for mulhi...)
    # more generally, we can take the low bits of everything:
    ab1 = (ablo + ((abmi << sb) & WORD_MAX) + ((abhi << (2*sb)) & WORD_MAX)) & WORD_MAX
    # and if sb == 32, we have the impression we can forget abhi:
    if sb == 32:
        ab2 = (ablo + ((abmi << sb) & WORD_MAX)) & WORD_MAX
        return ab0 == ab1 and ab0 == ab2
    else:
        return ab0 == ab1

print("Checking a few mullo:")
for args in [(1,2,10),
             (14121, 51234, 32),
             (ZZ.random_element(2**63), ZZ.random_element(2**63), 32),
             (2**63 - 1, 2**63 - 1, 32),
             ]:
    print(*args, check_split_mullo(*args))
print("=================")

# --> still, for best performance, check literature/google/existing software, and get inspiration from them

# for mulhi, this is trickier: the value of mulhi depends on whether there was or was not the overflow mentioned above... so it seems we need to compute the low part in order to get the high part (this is the problem of the possible carry)
