from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    id: str
    name: str
    role: str


USERS = [
    User("111", "Emma", "pm"),
    User("222", "Max", "rm"),
    User("333", "Paul", "admin"),
]

ROLE_LABELS = {
    "pm": "Portfolio Manager",
    "rm": "Risk Manager",
    "admin": "Admin",
}

PAGE_ACCESS = {
    "suitability": {"admin", "rm"},
    "strategies": {"admin", "pm"},
    "analytics": {"admin", "rm", "pm"},
    "admin": {"admin"},
}

PAGES = {
    "strategies": "Strategies",
    "analytics": "Analytics",
    "suitability": "Suitability",
    "admin": "Admin",
}


def get_user(user_id: str) -> User:
    return next(user for user in USERS if user.id == user_id)


def user_options() -> dict[str, str]:
    return {user.name: user.id for user in USERS}


def can_access(user: User, page: str) -> bool:
    return user.role in PAGE_ACCESS[page]
