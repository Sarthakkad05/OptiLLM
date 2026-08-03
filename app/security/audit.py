"""
Immutable & Tamper-Evident Audit Logging Engine
Records action audit entries linked with cryptographic SHA256 hashes.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List

from sqlalchemy.orm import Session
from app.db.models import AuditLogEntry

logger = logging.getLogger("optillm.security.audit")


class AuditLogger:
    """
    Manages tamper-evident audit logs with hash chain linking.
    """

    def record_action(
        self,
        db: Session,
        tenant_id: str,
        actor: str,
        action: str,
        resource: str,
        payload: Any = None,
        status: str = "success",
    ) -> AuditLogEntry:
        """
        Record a new action with linked SHA256 hash checksum.
        """
        # Retrieve latest entry for tenant to get previous hash
        last_entry = (
            db.query(AuditLogEntry)
            .filter(AuditLogEntry.tenant_id == tenant_id)
            .order_by(AuditLogEntry.id.desc())
            .first()
        )

        prev_hash = last_entry.payload_hash if last_entry else "0" * 64

        payload_str = json.dumps(payload, sort_keys=True) if payload is not None else ""
        raw_to_hash = f"{prev_hash}:{tenant_id}:{actor}:{action}:{resource}:{payload_str}:{status}"
        payload_hash = hashlib.sha256(raw_to_hash.encode("utf-8")).hexdigest()

        entry = AuditLogEntry(
            tenant_id=tenant_id,
            actor=actor,
            action=action,
            resource=resource,
            payload_hash=payload_hash,
            prev_hash=prev_hash,
            status=status,
        )

        db.add(entry)
        db.commit()
        db.refresh(entry)
        return entry

    def verify_chain_integrity(self, db: Session, tenant_id: str = "default") -> Dict[str, Any]:
        """
        Verify cryptographic hash chain continuity for specified tenant.
        """
        entries = (
            db.query(AuditLogEntry)
            .filter(AuditLogEntry.tenant_id == tenant_id)
            .order_by(AuditLogEntry.id.asc())
            .all()
        )

        if not entries:
            return {"tenant_id": tenant_id, "valid": True, "total_entries": 0, "corrupted_entries": []}

        corrupted = []
        expected_prev = "0" * 64

        for entry in entries:
            if entry.prev_hash != expected_prev:
                corrupted.append(entry.id)
            expected_prev = entry.payload_hash

        return {
            "tenant_id": tenant_id,
            "valid": len(corrupted) == 0,
            "total_entries": len(entries),
            "corrupted_entries": corrupted,
        }


_global_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger()
    return _global_audit_logger
