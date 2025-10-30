// Checkout Form Handler
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('checkout-form');
    const submitBtn = document.getElementById('submit-btn');
    const fraudStatus = document.getElementById('fraud-status');
    const resultModal = new bootstrap.Modal(document.getElementById('resultModal'));
    
    // Form validation
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        if (validateForm()) {
            processTransaction();
        }
    });
    
    // Real-time amount validation
    // Real-time amount validation
document.getElementById('amount').addEventListener('input', function() {
    const amount = parseFloat(this.value);
    const feedback = this.parentNode.parentNode.querySelector('.form-text');

    if (amount > 10000) {
        this.classList.remove('is-invalid');
        this.classList.add('is-valid');
        feedback.textContent = '⚠ High-value transaction - Enhanced security checks will apply';
        feedback.classList.remove('text-danger');
        feedback.classList.add('text-warning');
    } else {
        this.classList.remove('is-invalid', 'is-valid');
        feedback.textContent = 'Enter any transaction amount';
        feedback.classList.remove('text-danger', 'text-warning');
    }
});


    // Email validation
    // Email validation
document.getElementById('email').addEventListener('blur', function() {
    const email = this.value.toLowerCase();
    const feedback = this.parentNode.querySelector('.form-text');

    // Check for suspicious patterns
    const disposableDomains = ['10minutemail.com', 'tempmail.org', 'guerrillamail.com'];
    const domain = email.split('@')[1];

    if (disposableDomains.includes(domain)) {
        // Keep the warning but do not block submission
        this.classList.remove('is-invalid');
        this.classList.add('is-warning'); // optional: create a yellow border style in CSS
        feedback.textContent = '⚠ Disposable email addresses are not recommended for security';
        feedback.classList.add('text-warning');
    } else {
        this.classList.remove('is-invalid', 'is-warning');
        this.classList.add('is-valid');
        feedback.textContent = "We'll send confirmation to this email";
        feedback.classList.remove('text-warning');
    }
});

    
    function validateForm() {
        let isValid = true;
        const requiredFields = ['name', 'email', 'address', 'billing_country', 'payment_method', 'amount'];
        
        requiredFields.forEach(fieldName => {
            const field = document.getElementById(fieldName);
            const value = field.value.trim();
            
            if (!value) {
                field.classList.add('is-invalid');
                isValid = false;
            } else {
                field.classList.remove('is-invalid');
                field.classList.add('is-valid');
            }
        });
        
        // Email format validation
        const email = document.getElementById('email').value;
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            document.getElementById('email').classList.add('is-invalid');
            isValid = false;
        }
        
        const amount = parseFloat(document.getElementById('amount').value);
if (isNaN(amount) || amount <= 0) {
    document.getElementById('amount').classList.add('is-invalid');
    isValid = false;
} else {
    document.getElementById('amount').classList.remove('is-invalid');
    document.getElementById('amount').classList.add('is-valid');
}

        
        return isValid;
    }
    
    async function processTransaction() {
        submitBtn.innerHTML = '<div class="spinner-border spinner-border-sm me-2"></div>Processing...';
        submitBtn.disabled = true;
        fraudStatus.style.display = 'block';
        form.classList.add('loading');
        
        // Collect form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        data.billing_country = document.getElementById('billing_country').value;
        
        try {
            const response = await fetch('/api/process_transaction', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (result.success) {
                showTransactionResult(result);
                form.reset();
                form.querySelectorAll('.is-valid').forEach(el => el.classList.remove('is-valid'));
            } else {
                throw new Error(result.error || 'Transaction failed');
            }
            
        } catch (error) {
            showError(error.message);
        } finally {
            submitBtn.innerHTML = '<i class="bi bi-shield-check me-2"></i>Process Secure Payment';
            submitBtn.disabled = false;
            fraudStatus.style.display = 'none';
            form.classList.remove('loading');
        }
    }
    
    function showTransactionResult(result) {
        const modalHeader = document.getElementById('modal-header');
        const modalTitle = document.getElementById('modal-title');
        const modalBody = document.getElementById('modal-body');
        let statusClass, statusIcon, statusText;
        
        if (result.decision === 'Approved') {
    statusClass = 'bg-success';
    statusIcon = 'bi-check-circle';
    statusText = 'Transaction Approved';
} else if (result.decision === 'Under Review') {
    statusClass = 'bg-warning';
    statusIcon = 'bi-exclamation-triangle';
    statusText = 'Transaction Under Review';
} else {
    statusClass = 'bg-secondary';
    statusIcon = 'bi-question-circle';
    statusText = 'Status Unknown';
}

        modalHeader.className = `modal-header ${statusClass} text-white`;
        modalTitle.innerHTML = `<i class="bi ${statusIcon} me-2"></i>${statusText}`;
        
        const scoreColor = result.fraud_score <= 40 ? 'success' : 
                          result.fraud_score <= 80 ? 'warning' : 'danger';
        
        modalBody.innerHTML = `
            <div class="text-center mb-4">
                <div class="fraud-score-circle mx-auto mb-3" style="width: 120px; height: 120px; position: relative;">
                    <svg width="120" height="120" class="rotate-90">
                        <circle cx="60" cy="60" r="50" fill="none" stroke="#e9ecef" stroke-width="8"/>
                        <circle cx="60" cy="60" r="50" fill="none" stroke="var(--bs-${scoreColor})" 
                                stroke-width="8" stroke-dasharray="314" 
                                stroke-dashoffset="${314 - (314 * result.fraud_score / 100)}"
                                stroke-linecap="round" style="transition: stroke-dashoffset 1s ease;"/>
                    </svg>
                    <div class="position-absolute top-50 start-50 translate-middle text-center">
                        <div class="h4 mb-0 text-${scoreColor}">${result.fraud_score}</div>
                        <small class="text-muted">Risk Score</small>
                    </div>
                </div>
                <h5 class="text-${scoreColor}">${result.message}</h5>
            </div>
            
            <div class="row mb-4">
                <div class="col-md-6">
                    <h6><i class="bi bi-receipt me-2"></i>Transaction Details</h6>
                    <ul class="list-unstyled">
                        <li><strong>ID:</strong> #${result.transaction_id}</li>
                        <li><strong>Status:</strong> 
    <span class="badge ${result.decision === 'Approved' ? 'bg-success' : 'bg-warning'}">
        ${result.decision}
    </span>
</li>

                        <li><strong>Risk Score:</strong> 
                            <span class="fraud-score-${scoreColor}">${result.fraud_score}/100</span>
                        </li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <h6><i class="bi bi-info-circle me-2"></i>Risk Analysis</h6>
                    <div class="small">
                        <strong>Primary Factor:</strong><br>
                        ${result.explanation.top_reason}
                    </div>
                </div>
            </div>
            
            <div class="alert alert-info">
                <h6><i class="bi bi-lightbulb me-2"></i>Security Analysis</h6>
                <div class="small">
                    ${result.explanation.detailed_explanation.replace(/\n/g, '<br>')}
                </div>
            </div>
        `;
        
        resultModal.show();
        setTimeout(() => {
            const circle = modalBody.querySelector('circle:last-child');
            if (circle) circle.style.strokeDashoffset = `${314 - (314 * result.fraud_score / 100)}`;
        }, 300);
    }
    
    function showError(message) {
        const modalHeader = document.getElementById('modal-header');
        const modalTitle = document.getElementById('modal-title');
        const modalBody = document.getElementById('modal-body');
        
        modalHeader.className = 'modal-header bg-danger text-white';
        modalTitle.innerHTML = '<i class="bi bi-x-circle me-2"></i>Transaction Failed';
        
        modalBody.innerHTML = `
            <div class="alert alert-danger">
                <h6><i class="bi bi-exclamation-triangle me-2"></i>Error</h6>
                <p class="mb-0">${message}</p>
            </div>
            <p class="text-muted">Please check your information and try again. If the problem persists, contact support.</p>
        `;
        
        resultModal.show();
    }
});

// Utility functions
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(amount);
}

// Auto-format phone number
document.getElementById('phone').addEventListener('input', function() {
    let value = this.value.replace(/\D/g, '');
    if (value.length >= 6) {
        value = value.replace(/(\d{3})(\d{3})(\d{4})/, '($1) $2-$3');
    } else if (value.length >= 3) {
        value = value.replace(/(\d{3})(\d{3})/, '($1) $2');
    }
    this.value = value;
});
