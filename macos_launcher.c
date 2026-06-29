#include <stdio.h>
#include <time.h>
#include <unistd.h>

int main(void) {
    chdir("/Users/shogohayashi/voice-input-tool");

    FILE *launcher_log = fopen("/Users/shogohayashi/voice-input-tool/logs/app-launcher.log", "a");
    if (launcher_log) {
        time_t now = time(NULL);
        fprintf(launcher_log, "launcher started at %ld\n", (long)now);
        fflush(launcher_log);
        fclose(launcher_log);
    }

    freopen("/Users/shogohayashi/voice-input-tool/logs/app-launcher.log", "a", stdout);
    freopen("/Users/shogohayashi/voice-input-tool/logs/app-launcher-error.log", "a", stderr);

    char *const args[] = {
        "/Users/shogohayashi/voice-input-tool/.venv-framework/bin/python3",
        "/Users/shogohayashi/voice-input-tool/voice_input.py",
        "--llm",
        NULL,
    };

    execv(args[0], args);
    perror("execv");
    return 1;
}
