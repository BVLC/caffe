#ifndef INCLUDE_COMPAT_SIGNAL_COMPAT_H_
#define INCLUDE_COMPAT_SIGNAL_COMPAT_H_

#ifdef _MSC_VER

#define SIGHUP SIGBREAK

#define SA_NOCLDSTOP 1
#define SA_NOCLDWAIT 2
#define SA_SIGINFO   4
#define SA_ONSTACK   0x08000000
#define SA_RESTART   0x10000000
#define SA_INTERRUPT 0x20000000
#define SA_NODEFER   0x40000000
#define SA_RESETHAND 0x80000000

#define SA_NOMASK    SA_NODEFER
#define SA_ONESHOT   SA_RESETHAND
#define SA_STACK     SA_ONSTACK

#define SIG_NOP     0
#define SIG_BLOCK   1
#define SIG_UNBLOCK 2
#define SIG_SETMASK 3

typedef unsigned long sigset_t;  // NOLINT(runtime/int)

typedef struct sigaction {
    void(*sa_handler)(int);
    sigset_t sa_mask;
    int sa_flags;
} sigaction_t;

int sigaction(int, struct sigaction *, struct sigaction *);
int sigsuspend(sigset_t *set);
int sigprocmask(int op, sigset_t *set, sigset_t *oset);
int sigemptyset(sigset_t *set);
int sigfillset(sigset_t *set);

#endif

#endif  // INCLUDE_COMPAT_SIGNAL_COMPAT_H_
