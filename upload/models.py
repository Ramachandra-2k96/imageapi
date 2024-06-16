from django.db import models
from django.contrib.auth.models import User

class CreditCardTransaction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    account_number=models.IntegerField()
    merchant = models.CharField(max_length=100)
    Chequenumber = models.CharField(max_length=16)
    Account_holder_name = models.CharField(max_length=100)
    signature_image = models.ImageField(upload_to='signatures/')
    
    def __str__(self):
        return f"Transaction {self.id} - {self.user.username}"
