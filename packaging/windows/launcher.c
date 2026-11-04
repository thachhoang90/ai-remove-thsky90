/* AI Remove - thsky90 : tiny launcher.
 * Starts runtime\pythonw.exe app\main.py (portable layout next to this exe). */
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <wchar.h>
#include <stdio.h>

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, LPWSTR args, int nShow) {
    wchar_t dir[MAX_PATH];
    DWORD n = GetModuleFileNameW(NULL, dir, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return 1;
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (slash) *slash = 0;

    wchar_t py[MAX_PATH + 64], script[MAX_PATH + 64];
    _snwprintf(py, MAX_PATH + 64, L"%ls\\runtime\\pythonw.exe", dir);
    _snwprintf(script, MAX_PATH + 64, L"%ls\\app\\main.py", dir);
    if (GetFileAttributesW(py) == INVALID_FILE_ATTRIBUTES) {
        MessageBoxW(NULL, L"Thiếu thư mục runtime. Hãy cài đặt lại AI Remove - thsky90.",
                    L"AI Remove - thsky90", MB_ICONERROR);
        return 2;
    }
    size_t len = wcslen(py) + wcslen(script) + (args ? wcslen(args) : 0) + 16;
    wchar_t *cmd = (wchar_t *)HeapAlloc(GetProcessHeap(), 0, len * sizeof(wchar_t));
    _snwprintf(cmd, len, L"\"%ls\" \"%ls\" %ls", py, script, args ? args : L"");

    SetEnvironmentVariableW(L"PYTHONHOME", NULL);
    SetEnvironmentVariableW(L"PYTHONPATH", NULL);
    SetEnvironmentVariableW(L"PYTHONIOENCODING", L"utf-8");
    SetEnvironmentVariableW(L"QT_ENABLE_HIGHDPI_SCALING", L"1");

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));
    if (!CreateProcessW(py, cmd, NULL, NULL, FALSE, 0, NULL, dir, &si, &pi)) {
        wchar_t msg[512];
        _snwprintf(msg, 512, L"Không khởi động được (mã lỗi %lu).", GetLastError());
        MessageBoxW(NULL, msg, L"AI Remove - thsky90", MB_ICONERROR);
        return 3;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return 0;
}
