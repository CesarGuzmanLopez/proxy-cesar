---
applyTo: "**/*.py"
name: "Python Backend Guide - FastAPI Moderno"
description: "Guía integral para generar código tipado, modular, mantenible, alineado con Python 3.14+ y FastAPI usando arquitectura hexagonal"
---

# GUÍA BACKEND (PYTHON + FASTAPI MODERNO)

## 0. VERSIONES OBLIGATORIAS

| Componente    | Versión mínima          |
| ------------- | ----------------------- |
| Python        | **3.14+**               |
| FastAPI       | **0.136+**              |
| SQLModel      | **0.0.38+**             |

---

## 1. ARQUITECTURA HEXAGONAL — OBLIGATORIA

### 1.1 Capas

| Capa            | Responsabilidad                            |
| --------------- | ------------------------------------------ |
| `domain/`       | Entidades puras, sin dependencia de FastAPI/SQLModel |
| `services/`     | Lógica de aplicación                       |
| `ports/`        | Interfaces (Protocols/ABCs)                |
| `adapters/`     | Implementaciones (DB, API, cache)          |
| `routers/`      | Endpoints FastAPI (presentación)           |
| `schemas/`      | Pydantic/SQLModel schemas                  |

**Regla**: el dominio NO importa FastAPI ni SQLModel. FastAPI es infraestructura.

### 1.2 Estructura

```
src/backend/apps/<feature>/
  domain/     services/     ports/
  adapters/   schemas/      routers/
```

### 1.3 Estructura plana alternativa (features pequeñas)

```
src/backend/<feature>/
  __init__.py       # router = APIRouter(...)
  domain.py         # entidades
  service.py        # lógica de aplicación
  schemas.py        # Pydantic request/response
  repository.py     # SQLModel adapter
  tests.py          # tests unitarios
```

Elegir según complejidad del módulo.

### 1.4 Tamaño de archivos

- Ideal: **300–400 líneas** — Mínimo: 100 — Máximo: **600** (PROHIBIDO superarlo)
- Dividir por responsabilidad, no sobre-fragmentar
- Archivos pequeños (< 100 líneas) se pueden fusionar por afinidad

---

## 2. TIPADO ESTRICTO — PRIORIDAD MÁXIMA

### 2.1 Reglas absolutas

| Prohibido               | Usar en cambio                         |
| ----------------------- | -------------------------------------- |
| `Any`                   | tipos concretos, `TypeVar`, `Protocol` |
| `list` sin parametrizar | `list[T]`                              |
| `dict` sin parametrizar | `dict[K, V]`, `TypedDict`              |
| `tuple` sin estructura  | `tuple[T1, T2]` o `tuple[T, ...]`      |
| `Callable` sin firma    | `Callable[[Args], Return]`             |
| `object` genérico       | `Protocol` o `TypeVar`                 |
| `# type: ignore`        | corregir el tipo o refactorizar        |
| `cast` excesivo         | validar desde el origen                |
| `NOSONAR` / supresiones | corregir el error                      |

### 2.2 Python 3.14+ — features obligatorias

| Feature                            | Uso                                       |
| ---------------------------------- | ----------------------------------------- |
| `X \| Y`                           | uniones (reemplaza `Union[X, Y]`)         |
| `type UserId = int`                | alias de tipos (PEP 695)                  |
| `def f[T](x: T) -> T`              | genéricos modernos (PEP 695)              |
| `Self`                             | referencia al tipo propio                 |
| `@override`                        | indicar sobreescritura                    |
| `TypedDict`                        | estructuras dict tipadas                  |
| `dataclass(slots=True)`            | dataclasses eficientes                    |
| `match/case`                       | control de flujo estructural              |
| `ReadOnly` / `ReadOnly` (3.13+)    | campos TypedDict de solo lectura          |
| `TypeIs` (3.14)                    | narrowing de tipos más preciso            |
| `warnings.deprecated` (3.13)       | deprecar con soporte estático             |
| Anotaciones diferidas (3.14)       | sin `from __future__ import annotations`  |
| `types.UnionType` unificado (3.14) | `int \| str` equivale a `Union[int, str]` |

```python
type UserId = int

class UserData(TypedDict):
    id: UserId
    name: str
    email: ReadOnly[str]

def find_user[T: BaseUser](repo: UserPort[T], user_id: UserId) -> T | None:
    ...
```

---

## 3. MÓNADAS PARA MANEJO DE ERRORES — OBLIGATORIO

### 3.1 Patrón Result

No usar excepciones como flujo principal de lógica de negocio. Representar errores como datos:

```python
# types.py
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")

@dataclass(frozen=True, slots=True)
class Ok[T]:
    value: T
    ok: bool = True

@dataclass(frozen=True, slots=True)
class Err[E]:
    error: E
    ok: bool = False

type Result[T, E] = Ok[T] | Err[E]
```

```python
# Uso en servicio
def create_product(name: str, price: float) -> Result[Product, ValidationError]:
    if price < 0:
        return Err(ValidationError(f"Precio inválido: {price}"))
    return Ok(Product(name=name, price=price))

# Consumo en router
@router.post("/products")
def create_product_endpoint(data: ProductCreate) -> Product | ErrorResponse:
    result = create_product(data.name, data.price)
    match result:
        case Ok(value=product):
            repo.save(product)
            return product
        case Err(error=err):
            raise HTTPException(status_code=422, detail=str(err))
```

### 3.2 Reglas

- Retornar `Result[T, E]` en servicios y puertos — nunca `None` ambiguo
- Usar `match/case` para extraer valores, no condicionales anidados
- Propagar errores explícitamente, no silenciarlos
- Excepciones solo para errores inesperados de infraestructura (IO, DB)
- `HTTPException` solo en routers, nunca en services/domain

---

## 4. PROGRAMACIÓN FUNCIONAL

### 4.1 Principios

| Regla           | Descripción                                    |
| --------------- | ---------------------------------------------- |
| Funciones puras | sin efectos secundarios ocultos                |
| Inmutabilidad   | `dataclass(frozen=True)`, `tuple`, `frozenset` |
| Composición     | encadenar en lugar de anidar `if`              |
| Declarativo     | describir QUÉ, no CÓMO                         |

```python
def process_products(prices: list[float]) -> list[Result[Product, str]]:
    return [create_product(f"Producto {i}", p) for i, p in enumerate(prices)]

valid = [r.value for r in results if isinstance(r, Ok)]
```

### 4.2 Funciones

- Máx **20–30 líneas**, una responsabilidad
- Siempre tipadas: parámetros + retorno
- Sin efectos secundarios en funciones de dominio

---

## 5. PATRONES DE DISEÑO Y POO — OBLIGATORIO

### 5.1 Principios OOP

| Principio         | Aplicación en FastAPI/Python                       |
| ----------------- | -------------------------------------------------- |
| Encapsulamiento   | atributos privados `_name`, propiedades controladas |
| Abstracción       | `Protocol` / ABC para puertos                      |
| SRP               | un archivo = una responsabilidad clara              |
| LSP               | subclases sustituibles, no romper contratos         |
| Bajo acoplamiento | depender de abstracciones, no implementaciones      |

### 5.2 Patrones

| Patrón         | Cuándo usarlo               | Implementación                                  |
| -------------- | --------------------------- | ----------------------------------------------- |
| **Repository** | acceso a datos              | `Protocol` en ports, SQLModel en adapters       |
| **Factory**    | crear objetos complejos     | función/clase que retorna `Result[T, E]`        |
| **Adapter**    | integrar librerías externas | wrapper en `adapters/` con interfaz propia      |
| **Facade**     | simplificar subsistemas     | servicio que orquesta múltiples ports           |
| **Strategy**   | algoritmos intercambiables  | `Protocol` con `__call__` o método único        |
| **DI**         | inyección de dependencias   | parámetros en `__init__` + `Depends()` de FastAPI |

```python
# Repository — Puerto
class ProductRepository(Protocol):
    def find_by_id(self, product_id: int) -> Product | None: ...
    def save(self, product: Product) -> Result[Product, str]: ...

# Adapter — SQLModel
class SQLModelProductRepository:
    def find_by_id(self, product_id: int) -> Product | None:
        with Session(engine) as session:
            model = session.get(ProductModel, product_id)
            return model.to_domain() if model else None

    def save(self, product: Product) -> Result[Product, str]:
        with Session(engine) as session:
            model = ProductModel.from_domain(product)
            session.add(model)
            session.commit()
            return Ok(model.to_domain())
```

### 5.3 Inyección de dependencias con FastAPI

```python
# adapters/
def get_repository() -> ProductRepository:
    return SQLModelProductRepository(engine)

# routers/
@router.get("/products/{product_id}")
def get_product(
    product_id: int,
    repo: ProductRepository = Depends(get_repository),
    service: ProductService = Depends(get_product_service),
) -> Product | ErrorResponse:
    result = service.get_product(repo, product_id)
    ...
```

---

## 6. FASTAPI MODERNO

### 6.1 Organización de routers

```python
# src/backend/products/routers.py
from fastapi import APIRouter

router = APIRouter(prefix="/products", tags=["products"])

@router.get("/", response_model=list[ProductRead])
def list_products(repo: ProductRepository = Depends(get_repository)):
    ...

@router.post("/", response_model=ProductRead, status_code=201)
def create_product(
    data: ProductCreate,
    repo: ProductRepository = Depends(get_repository),
):
    ...

# src/backend/main.py
from backend.products.routers import router as products_router

app = FastAPI(title="Mi API", version="1.0.0")
app.include_router(products_router)
```

### 6.2 SQLModel — modelos

```python
from sqlmodel import SQLModel, Field
from datetime import datetime

class ProductModel(SQLModel, table=True):
    __tablename__ = "products"
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    price: float
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_domain(self) -> Product:
        return Product(id=self.id, name=self.name, price=self.price)

    @classmethod
    def from_domain(cls, product: Product) -> "ProductModel":
        return cls(name=product.name, price=product.price)
```

### 6.3 Pydantic v2 — schemas de request/response

```python
from pydantic import BaseModel, Field

class ProductCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    price: float = Field(gt=0)

class ProductRead(BaseModel):
    id: int
    name: str
    price: float
```

### 6.4 Validación

- Validación de formato/input → Pydantic (`Field`, `field_validator`)
- Validación de negocio → services (con `Result[T, E]`)
- Validación DB → constraints SQLModel/DB
- NO duplicar validaciones

### 6.5 OpenAPI

FastAPI genera OpenAPI automáticamente. Solo añadir metadata extra cuando sea necesario:

```python
from fastapi import APIRouter, Query

@router.get("/products")
def list_products(
    q: str = Query("", description="Filtro de búsqueda"),
    page: int = Query(1, ge=1),
):
    ...
```

Generar schema con:
```bash
python generate_openapi.py
```

---

## 7. ASYNC — OBLIGATORIO

FastAPI es async-first por defecto. Preferir `async def` en routers y usar `AsyncSession` de SQLModel para DB:

```python
from sqlmodel.ext.asyncio.session import AsyncSession

async def get_product(
    product_id: int,
    repo: AsyncProductRepository = Depends(get_async_repository),
) -> Product | None:
    return await repo.find_by_id(product_id)

@router.get("/products/{product_id}")
async def get_product_endpoint(
    product_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    ...
```

- Usar `async def` en todos los endpoints
- Usar sesión asíncrona de SQLModel para operaciones DB
- Usar `httpx.AsyncClient` para llamadas HTTP externas
- Usar `asyncio.to_thread` para operaciones blocking inevitables

---

## 8. ANTI-PATTERNS — NUNCA HACER

- `Any`, `object` genérico, `cast` sin necesidad
- Excepciones como control de flujo principal → usar `Result[T, E]`
- `except Exception` o `except:` vacío
- Lógica de negocio en routers o modelos
- Duplicación de validaciones (Pydantic + service + DB)
- Archivos > 600 líneas
- Instanciar dependencias con `new` directo dentro de clases → usar DI + `Depends()`
- `# type: ignore` sin justificación explícita en comentario
- `from __future__ import annotations` en Python 3.14+ (ya no necesario)
- `Union[X, Y]` en lugar de `X | Y`
- Poner lógica pesada en el startup de la app
- Dependencias circulares entre routers

---

## 9. ESTRUCTURA COMPLETA RECOMENDADA

```
backend/
  pyproject.toml
  generate_openapi.py
  openapi/
    schema.json
  src/backend/
    __init__.py
    main.py                        # app = FastAPI(...), include routers
    database.py                    # engine, session factory
    config.py                      # settings from env vars
    apps/
      products/
        __init__.py                # router global del módulo
        domain.py                  # entidades
        schemas.py                 # Pydantic request/response
        service.py                 # lógica
        repository.py              # SQLModel adapter
        tests.py                   # tests unitarios
      users/
        ...
  tests/
    conftest.py                    # fixtures de DB, client, etc.
    test_products.py
    test_users.py
```

---

## REGLA META — ORDEN DE PRIORIDADES

1. **Tipado completo** — `Result[T, E]`, sin `Any`
2. **Arquitectura hexagonal** — dominio puro, ports/adapters
3. **Patrones de diseño y encapsulamiento**
4. **Manejo de errores con mónadas** — `Result` en lógica de negocio
5. **Async-first** — `async def` + AsyncSession
6. **Modularidad** — archivos 300–400 líneas
7. **Resto de convenciones**

> Reglas ignorables SOLO si se justifica con comentario en el código explicando por qué y cómo mejora la claridad. Los archivos autogenerados en `**/generated/**` se excluyen de todas las reglas.
>
> Si mónadas o abstracciones funcionales reducen la legibilidad, preferir solución más simple con justificación comentada.
