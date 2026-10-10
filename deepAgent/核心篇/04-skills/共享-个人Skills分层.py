from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend


@dataclass(frozen=True)
class TenantContext:
    org_id: str
    user_id: str

# 共享skills
def shared_skill_namespace(rt):
    org_id = getattr(rt.context, "org_id", "default-org")
    return ("curated-skills", org_id)

def personal_skill_namespace(rt):
    if rt.server_info and rt.server_info.user:
        return ("user-skills", rt.server_info.user.identity)
    user_id = getattr(rt.context, "user_id", "local-user")
    return ("user-skills", user_id)

agent = create_deep_agent(
    model="deepseek:deepseek-flash",
    context_schema=TenantContext,
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/skills/shared/": StateBackend(
                namespace=shared_skill_namespace,
            ),
            "/skills/personal/": StateBackend(
                namespace=personal_skill_namespace,
            ),
        },
    ),
    skills=["/skills/shared/", "/skills/personal/"],
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["skills/shared/**"],
            mode="deny",
        )
    ]
)

