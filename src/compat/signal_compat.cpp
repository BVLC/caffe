#include <compat/signal_compat.h>
#include <signal.h>

#ifdef _MSC_VER

int sigaction(int sig, struct sigaction *sa, struct sigaction *osa) {
    if (osa)
        osa->sa_handler = signal(sig,
            reinterpret_cast<void(__cdecl*)(int)>(sa->sa_handler));  // NOLINT - false positive
    else
        signal(sig, reinterpret_cast<void(__cdecl*)(int)>(sa->sa_handler));  // NOLINT - false positive
    return 0;
}

int sigsuspend(sigset_t *set) {
    return 0;
}

int sigprocmask(int op, sigset_t *set, sigset_t *oset) {
    if (oset) *oset = 0;
    return 0;
}

int sigemptyset(sigset_t *set) {
    return *set = 0;
}

int sigfillset(sigset_t *set) {
    *set = ~(sigset_t)0; return 0;
}

#endif

