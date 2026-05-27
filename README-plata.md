# README-plata — Reglas del Servidor de Producción

> **Host:** `plata` (Ubuntu 22.04)
> **IP/dominio:** `chat.guzman-lopez.com` → Caddy → `localhost:9110`
> **Propietario del proxy:** usuario `proxy`

---

## ⚠️ REGLAS CRÍTICAS — NO VIOLAR

### 1. NUNCA USAR PIP FUERA DE VIRTUALENV

```bash
# MAL — NUNCA:
pip install <paquete>

# BIEN — siempre:
python3 -m venv .venv && source .venv/bin/activate
pip install <paquete>
```

### 2. NO TOCAR PAQUETES DEL SISTEMA

`/usr/lib/python3/dist-packages/` contiene **certbot** y dependencias del sistema. Modificar esto rompe la renovación de SSL y puede dejar el servidor sin HTTPS.

```bash
# MAL:
pip install --system <paquete>
sudo pip install <paquete>

# BIEN:
# Usar solo virtualenvs en /home/proxy/ o /tmp/
```

### 3. El proxy corre como usuario `proxy`

- Servicio systemd: `proxy-cesar.service`
- Usuario: `proxy` (hardcoded, no dinámico)
- Directorio: `/tmp/proxy-cesar/proxy/`
- No hacer `chown` a root después del deploy

### 4. No romper otros proyectos

El servidor aloja:

| Proyecto | Tipo | Puertos | Cuenta |
|---|---|---|---|
| **proxy-cesar** | FastAPI systemd | :9110 | `proxy` |
| **deepbde** | Docker compose | :8000, :6379 (Redis Docker) | `root` |
| **chemistry-apps** | Docker compose | :8080, :4210 | `root` |
| **Moodle** | PHP/Apache | :80→Caddy | — |

**Redis nativo del proxy** está en **puerto 6380** (systemd `redis-6380.service`).
**Redis de deepbde** está en **puerto 6379** (Docker `deepbde-redis`).
**NO** interferir con los Redis del otro.

### 5. PostgreSQL 14 existe pero el proxy NO lo usa

PostgreSQL 14 corre en `localhost:5432`. Está disponible para otros proyectos.
El proxy usa **SQLite** (archivo `/tmp/proxy-cesar/proxy/proxy.db`).

### 6. La DB SQLite se preserva entre deploys

El `deploy.yml` hace backup antes de clonar y restore después. **No borrar manualmente.**

---

## Servicios del Sistema

```bash
# Proxy
systemctl status proxy-cesar       # FastAPI :9110
journalctl -u proxy-cesar -n 50    # Logs

# Redis nativo (proxy)
systemctl status redis-6380         # :6380

# Caddy (reverse proxy)
systemctl status caddy              # HTTPS → :9110

# Otros proyectos (no tocar)
docker ps                           # deepbde, chemistry-apps
```

## Rutas importantes

| Ruta | Propósito |
|---|---|
| `/tmp/proxy-cesar/` | Código del proxy (git clone) |
| `/tmp/proxy-cesar/proxy/proxy.db` | SQLite DB |
| `/tmp/proxy-cesar/proxy/.env` | API keys |
| `/etc/systemd/system/proxy-cesar.service` | Service unit |
| `/etc/caddy/Caddyfile` | Config de Caddy |
| `/etc/redis/redis-6380.conf` | Config de Redis nativo |

## Backup y Restore Manual de DB

```bash
# Backup
cp /tmp/proxy-cesar/proxy/proxy.db /tmp/proxy.db.bak

# Restore
cp /tmp/proxy.db.bak /tmp/proxy-cesar/proxy/proxy.db
chown proxy:proxy /tmp/proxy-cesar/proxy/proxy.db
```

## Caddyfile (proxy routing)

```caddy
chat.guzman-lopez.com {
    reverse_proxy localhost:9110
}
```

---

> **Última actualización:** Mayo 2026
> **Mantenido por:** César Guzmán
