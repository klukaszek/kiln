[shader("compute")]
[numthreads(CLEAR_THREADS, 1, 1)]
void clearMain(uint3 tid : SV_DispatchThreadID, uniform ClearRoot* r)
{
    if (tid.x < r.count) {
        r.film[tid.x] = 0.0;
    }
}
