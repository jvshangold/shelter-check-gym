from dataclasses import dataclass, field
from typing import Dict, Optional, Literal

Jurisdiction = Literal["Ireland", "Netherlands", "Bermuda", "US", "Germany"]


@dataclass
class Entity:
    id: str
    incorporation_jurisdiction: Jurisdiction  # = operating location
    management_jurisdiction: Jurisdiction # = management location
    tax_residence: Jurisdiction
    parent_id: Optional[str] = None


@dataclass
class WorldState:
    entities: Dict[str, Entity] = field(default_factory=dict)
    subsidiary: set[tuple[str, str]] = field(default_factory=set)
    licenses: Dict[str, str] = field(default_factory=dict)  # licensee -> licensor
    ip_owner: Optional[str] = None
    _next_entity_id: int = 1

    country_revenue: Dict[Jurisdiction, float] = field(default_factory=lambda: {
        "US": 700_000_000.0,
        "Germany": 300_000_000.0,
        "Netherlands": 100_000_000.0,
        "Ireland": 30_000_000.0,
        "Bermuda": 0.0,
    })

    royalty_rate: float = 0.9

    def add_child(
        self,
        parent: str,
        incorporation_jurisdiction: Jurisdiction,
        management_jurisdiction: Jurisdiction,
        tax_residence: Jurisdiction,
    ) -> str:
        if parent not in self.entities:
            raise ValueError(f"Unknown parent entity: {parent}")

        child_id = f"company_{self._next_entity_id}"
        self._next_entity_id += 1

        child = Entity(
            id=child_id,
            incorporation_jurisdiction=incorporation_jurisdiction,
            management_jurisdiction=management_jurisdiction,
            tax_residence=tax_residence,
            parent_id=parent,
        )

        self.entities[child_id] = child
        self.subsidiary.add((parent, child_id))
        return child_id

    def transfer_ip(self, new_owner: str) -> None:
        if new_owner not in self.entities:
            raise ValueError(f"Unknown entity: {new_owner}")

        self.ip_owner = new_owner

    def rent_ip(self, licensee: str, licensor: str) -> None:
        if licensee not in self.entities:
            raise ValueError(f"Unknown licensee: {licensee}")
        if licensor not in self.entities:
            raise ValueError(f"Unknown licensor: {licensor}")
        if licensee == licensor:
            raise ValueError("A company cannot license IP from itself.")
        if self.ip_owner is None:
            raise ValueError("No company currently owns the IP.")

        # licensor must have rights (owner or already licensed)
        if not (licensor == self.ip_owner or licensor in self.licenses):
            raise ValueError(
                f"{licensor} does not own or have a license to use IP"
            )

        # cycle check
        cur = licensor
        while cur in self.licenses:
            cur = self.licenses[cur]
            if cur == licensee:
                raise ValueError("This would create a licensing cycle")

        self.licenses[licensee] = licensor
    
    def get_company_revenue(self, entity_id: str) -> float:
        if entity_id not in self.entities:
            raise ValueError(f"Unknown entity: {entity_id}")
    
        entity = self.entities[entity_id]
        country = entity.incorporation_jurisdiction

        eligible = [
            e_id
            for e_id, e in self.entities.items()
            if e.incorporation_jurisdiction == country and self.has_ip_rights(e_id)
        ]

        if entity_id not in eligible:
            return 0.0

        if not eligible:
            return 0.0

        return self.country_revenue[country] / len(eligible)
    

    def has_ip_rights(self, entity_id: str) -> bool:
        '''Helper to be used to divide revenue coming from a country'''
        return entity_id in self.licenses or entity_id == self.ip_owner
    
    def get_tax_residence(self, incorporation_jurisdiction, management_jurisdiction):
        eu_countries = {"Ireland", "Netherlands", "Germany"}

        if incorporation_jurisdiction in eu_countries:
            return incorporation_jurisdiction
        else:
            return management_jurisdiction
        

    def add_root(self,
                entity_id: Entity,
                incorporation_jurisdiction: Jurisdiction,
                management_jurisdiction: Jurisdiction,
                tax_residence: Jurisdiction):
        
        self.entities[entity_id] = Entity(incorporation_jurisdiction, 
                                            management_jurisdiction,
                                            tax_residence)
        self.ip_owner = entity_id
        return entity_id