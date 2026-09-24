export function readName() {
  try {
    return decodeURIComponent(
      document.cookie
        .split('; ')
        .find((s) => s.startsWith('md_username='))
        ?.slice(12) ?? '',
    );
  } catch {
    return '';
  }
}
export function saveName(name: string) {
  document.cookie = `md_username=${encodeURIComponent(name)}; Max-Age=31536000; Path=/; SameSite=Lax`;
}
export function readPreference(key: string, session = false) {
  try {
    return (session ? sessionStorage : localStorage).getItem(key) ?? '';
  } catch {
    return '';
  }
}
export function savePreference(key: string, value: string, session = false) {
  try {
    (session ? sessionStorage : localStorage).setItem(key, value);
  } catch {}
}
