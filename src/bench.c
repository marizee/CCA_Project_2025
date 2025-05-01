
void flint_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
{
    dot_params_t params = _nmod_vec_dot_params(len, mod);
    *res = _nmod_vec_dot(vec1, vec2, len, mod, params);
}
