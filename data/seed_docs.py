DOCUMENTS = [
    "Pour obtenir une carte d'identité nationale au Sénégal, présentez une copie de l'extrait de naissance, deux photos d'identité récentes et le reçu de paiement au guichet de la mairie de votre commune de résidence. Le délai de délivrance est généralement de 15 jours.",
    "Créer une entreprise au Sénégal nécessite la carte d'identité nationale, un extrait du registre de commerce et un justificatif de domicile. Les formalités se font auprès de l'APIX via le guichet unique situé boulevard du Centenaire à Dakar.",
    "L'extrait de naissance se demande à la mairie de votre lieu de naissance. Vous devez présenter le carnet de famille et la pièce d'identité du parent concerné. Le document est délivré au service de l'état civil.",
    "Pour un passeport biométrique, déposez le formulaire de demande dûment rempli et l'extrait de naissance au commissariat le plus proche ou à la direction de la police nationale. Les frais s'élèvent à 25000 francs CFA.",
    "La carte nationale d'identité est gratuite pour les personnes de plus de 60 ans. Pour les autres, les frais d'établissement s'élèvent à 5000 francs CFA, payables à la mairie.",
]

FILTERED_DOCS = [
    {"text": "La demande d'extrait de naissance se fait à la mairie de votre lieu de naissance, service de l'état civil. Rendez-vous au guichet n°3.",
     "meta": {"lieu": True}},
    {"text": "L'APIX gère les formalités de création d'entreprise. Contactez le guichet unique au boulevard du Centenaire à Dakar.",
     "meta": {"service": True, "lieu": True}},
    {"text": "Le passeport se retire à la direction de la police nationale, département des passeports, après convocation.",
     "meta": {"service": True, "lieu": True}},
    {"text": "Les frais de carte d'identité sont de 5000 FCFA (gratuit pour les plus de 60 ans).",
     "meta": {"cout": True}},
]


def get_documents():
    return DOCUMENTS, FILTERED_DOCS
